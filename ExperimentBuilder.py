import torch
import torch.nn.functional as F
import torch.nn as nn
from DatasetsUtil.DataLoaderWrap import DataLoaderWrap
from util.experiment_util import AverageMeter, interleave, de_interleave, accuracy, save_checkpoint
import time
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class ExperimentBuilder(nn.Module):
    def __init__(self, backbone, classifier_net, confidence_net, optimizer, scheduler, meta_optimizer, meta_scheduler, args, experiment_name, data_loader : DataLoaderWrap) -> None:
        super(ExperimentBuilder, self).__init__()
        self.experiment_name = experiment_name
        self.backbone = backbone
        self.classifer_net = classifier_net
        self.confidence_net = confidence_net
        self.data_loader = data_loader
        self.args = args
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.meta_optimizer = meta_optimizer
        self.meta_scheduler = meta_scheduler

        if torch.cuda.device_count() > 1 and args.use_gpu:
            self.device = torch.cuda.current_device()
            self.backbone.to(self.device)
            self.confidence_net.to(self.device)
            self.classifer_net.to(self.device)
            self.backbone = nn.DataParallel(module=self.backbone)
            self.confidence_net = nn.DataParallel(module=self.confidence_net)
            self.classifer_net = nn.DataParallel(module=self.classifer_net)
            print('Use Multi GPU', self.device)
        elif torch.cuda.device_count() == 1 and args.use_gpu:
            self.device = torch.cuda.current_device()
            print("device is ", self.device)
            # sends the model from the cpu to the gpu
            self.backbone.to(self.device)
            self.confidence_net.to(self.device)
            self.classifer_net.to(self.device)
            print('Use GPU', self.device)
        else:
            print("use CPU")
            self.device = torch.device('cpu')  # sets the device to be CPU
            print(self.device)

        self.experiment_folder = os.path.abspath(self.experiment_name)
        self.writer = SummaryWriter(self.experiment_folder)
        self.experiment_saved_models = os.path.abspath(
            os.path.join(self.experiment_folder, "saved_models"))
        # If experiment directory does not exist
        if not os.path.exists(self.experiment_folder):
            os.mkdir(self.experiment_folder)  # create the experiment directory
        if not os.path.exists(self.experiment_saved_models):
            # create the experiment saved models directory
            os.mkdir(self.experiment_saved_models)

        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.losses = AverageMeter()
        self.losses_x = AverageMeter()
        self.losses_u = AverageMeter()
        self.mask_probs = AverageMeter()
        self.pre_train_confidence_loss = AverageMeter()
        self.losses_u_weighted = AverageMeter()
        self.confidence_net_weights = []
        self.confidence_net_stds = []

    def neummann_approximation(self, v, f, w, i=3, alpha=.1):
        """Neumann Series Approximation to the Inverse Hessian.

        Args:
            v (Tuple[torch.Tensor]): Validation Loss Derivative wrt Model Parameters
            f (Tuple[torch.Tensor]): Training Loss Derivative wrt Model Parameters
            w (torch.Generator): Model Parameters
            device (int): The Device to Send the Computed Tensors To
            i (int, optional): Number of Terms in the Neumann Series. Defaults to 3.
            alpha (float, optional): [description]. Defaults to .1.

        Returns:
            (Tuple[torch.Tensor]): Inverse Hessian Approximation by Neumann Series Pre-Multiplied by (v)
        """
        p = v
        
        for j in range(i):
            grad = torch.autograd.grad(f, w(), grad_outputs=v, retain_graph=True)  # (L^2_t/dwdw)*(dv/dw)
            v = [v_ - alpha * g for v_, g in zip(v, grad)]
            # p += v (Typo in the arxiv version of the paper)
            p = [p_ + v_ for p_, v_ in zip(p, v)]

        return p

    def hypergradient(self, validation_loss, training_loss, lambda_, w):
        v1 = torch.autograd.grad(validation_loss, w(), retain_graph=True, allow_unused=True)
        d_train_d_w = torch.autograd.grad(training_loss, w(), create_graph=True)
        if self.args.approx == "numn":
            v2 = self.neummann_approximation(v1, d_train_d_w, w, i=self.args.num_neumann_terms)
        elif self.args.approx == "identity":
            v2 = v1
        v3 = torch.autograd.grad(d_train_d_w, lambda_(), grad_outputs=v2, retain_graph=True)
        # d_val_d_lambda = torch.autograd.grad(validation_loss, lambda_())
        #return [d - v for d, v in zip(d_val_d_lambda, v3)]
        return [-1*v for v in v3]

    def compute_batch_loss(self):
        self.backbone.train()
        self.classifer_net.train()
        inputs_x, targets_x = self.data_loader.get_labelled_batch()
        inputs_x, targets_x = inputs_x.to(self.device), targets_x.to(self.device)

        batch_size = inputs_x.shape[0]

        inputs_u, inputs_u_w, inputs_u_s = self.data_loader.get_unlabeled_batch()
        inputs_u_w, inputs_u_s = inputs_u_w.to(self.device), inputs_u_s.to(self.device)
        inputs = interleave(
            torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*self.args.mu+1)

        features = self.backbone(inputs)
        logits = self.classifer_net(features)
        logits = de_interleave(logits, 2*self.args.mu+1)
        logits_x = logits[:batch_size]
        logits_u_w, logits_u_s = logits[batch_size:].chunk(2)

        del logits

        Lx = torch.nn.CrossEntropyLoss()(logits_x, targets_x)

        pseudo_label = torch.softmax(logits_u_w.detach()/self.args.T, dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(self.args.threshold).float().to(self.device)

        Lu = F.cross_entropy(logits_u_s, targets_u, reduction='none')
        
        with torch.no_grad():
            features = self.backbone(inputs_u_w)
        u_weight = self.confidence_net(features)

        masked_lu = Lu * mask
        weighted_lu =  u_weight[:,0] * masked_lu
        weighted_lu = weighted_lu.mean()

        loss = Lx + weighted_lu

        return loss, Lx, weighted_lu, masked_lu.mean(), mask.mean(), u_weight.mean(), u_weight.std()

    def compute_val_loss(self):
        losses = []
        validation_accuracy = AverageMeter()
        for i in range(20):
            inputs = self.data_loader.val_data_x[0+(i*50):50+(i*50)].to(self.device)
            targets = self.data_loader.val_data_y[0+(i*50):50+(i*50)].to(self.device)
            with torch.no_grad():
                features = self.backbone(inputs)
            logits = self.classifer_net(features)
            losses.append(torch.nn.CrossEntropyLoss()(logits, targets))
            prec1_, prec5_ = accuracy(logits, targets, topk=(1,5))
            validation_accuracy.update(prec1_.item())
        return torch.mean(torch.stack(losses)), validation_accuracy.avg

    def train_step(self, pre_train = False):
        loss, Lx, weighted_lu, lu, mask, mwn_outputs_avg, mwn_outputs_std = self.compute_batch_loss()

        self.confidence_net_weights.append(mwn_outputs_avg.item())
        self.confidence_net_stds.append(mwn_outputs_std.item())

        self.backbone.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        self.losses.update(loss.item())
        self.losses_x.update(Lx.item())
        self.losses_u.update(lu.item())
        self.losses_u_weighted.update(weighted_lu.item())
        self.mask_probs.update(mask.item())
    
    def pretrain_confidence_network(self):
        end = time.time()
        
        meta_optimizer_pre = torch.optim.SGD(self.confidence_net.parameters(), lr=self.args.hyper_lr, momentum=0.9, weight_decay=5e-4)

        for epoch in range(1):
            if self.args.progress:
                p_bar = tqdm(range(self.args.pre_train_steps))
            for step in range(self.args.pre_train_steps):
                self.data_time.update(time.time() - end)
                
                self.confidence_net.train()
                inputs_u, inputs_u_w, inputs_u_s = self.data_loader.get_unlabeled_batch()
                inputs_x = inputs_u_w.to(self.device)
                with torch.no_grad():
                    features = self.backbone(inputs_x)
                logits = self.confidence_net(features)
                targets = torch.ones(inputs_u_w.shape[0]).to(self.device)
                Lx = torch.nn.MSELoss()(logits, targets)

                self.pre_train_confidence_loss.update(Lx.item())

                self.confidence_net.zero_grad()
                Lx.backward()
                meta_optimizer_pre.step()

                self.batch_time.update(time.time() - end)
                end=time.time()
                if self.args.progress:
                        p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. ".format(
                        epoch=epoch + 1,
                        epochs=50,
                        batch=step + 1,
                        iter=self.args.pre_train_steps,
                        data=self.data_time.avg,
                        bt=self.batch_time.avg,
                        loss=self.pre_train_confidence_loss.avg))
                p_bar.update()

                if self.args.debugging:
                    self.writer.add_scalar('pre_train_confidence_net/1.train_loss', self.pre_train_confidence_loss.avg, step)
            self.confidence_net.eval()

    def pre_train(self):
        end = time.time()
        self.backbone.train()

        if self.args.pre_train:
            for epoch in range(self.args.pre_train_epochs):
                if self.args.progress:
                    p_bar = tqdm(range(self.args.pre_train_steps))
                for step in range(self.args.pre_train_steps):
                    self.data_time.update(time.time() - end)
                    self.train_step(pre_train=True)
                    self.batch_time.update(time.time() - end)
                    end=time.time()

                    if self.args.progress:
                        p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}. ".format(
                        epoch=epoch + 1,
                        epochs=self.args.pre_train_epochs,
                        batch=step + 1,
                        iter=self.args.pre_train_steps,
                        lr=self.scheduler.get_last_lr()[0],
                        data=self.data_time.avg,
                        bt=self.batch_time.avg,
                        loss=self.losses.avg,
                        loss_x=self.losses_x.avg,
                        loss_u=self.losses_u.avg,
                        mask=self.mask_probs.avg))
                    p_bar.update()
                    if self.args.debugging:
                        self.writer.add_scalar('pre_train_epoch'+str(epoch)+'/1.train_loss', self.losses.avg, step)
                        self.writer.add_scalar('pre_train_epoch'+str(epoch)+'/2.train_loss_x', self.losses_x.avg, step)
                        self.writer.add_scalar('pre_train_epoch'+str(epoch)+'/3.train_loss_u', self.losses_u.avg, step)
                        self.writer.add_scalar('pre_train_epoch'+str(epoch)+'/4.mask', self.mask_probs.avg, step)


                if self.args.progress:
                    p_bar.close()
        print("########################################################################")
        print("Pre-Training Finished. Starting Meta-Learning Routine.")
        print("########################################################################")

    def test(self):
        batch_size = 128
        
        test_acc_1 = AverageMeter()
        test_acc_5 = AverageMeter()
        test_loss = AverageMeter()
        
        with torch.no_grad():
            batches = round(self.data_loader.test_data_x.shape[0]/batch_size)
            for i in range(batches):
                self.backbone.eval()
                inputs = self.data_loader.test_data_x[0+(i*batch_size):batch_size+(i*batch_size)].to(self.device)
                targets = self.data_loader.test_data_y[0+(i*batch_size):batch_size+(i*batch_size)].to(self.device)
                outputs = self.backbone(inputs)
                loss = torch.nn.CrossEntropyLoss()(outputs, targets)

                prec1_, prec5_ = accuracy(outputs, targets, topk=(1,5))
                
                test_acc_1.update(prec1_.item())
                test_acc_5.update(prec5_.item())
                test_loss.update(loss.item())

            return test_loss.avg, test_acc_1.avg, test_acc_5.avg

    def meta_update(self):
        if not self.args.freeze_meta:
            self.confidence_net.train()
            train_loss, Lx, weighted_lu, lu, mask, mwn_outputs_avg, mwn_outputs_std = self.compute_batch_loss()
            val_loss, val_acc = self.compute_val_loss()
            
            hyper_grads = self.hypergradient(val_loss, train_loss, self.confidence_net.parameters, self.backbone.parameters)

            self.meta_optimizer.zero_grad()
            for p, g in zip(self.confidence_net.parameters(), hyper_grads):
                p.grad = g
            self.meta_optimizer.step()
            self.meta_scheduler.step()
            self.confidence_net.eval()

            return val_loss, val_acc
        else:
            with torch.no_grad():
                val_loss, val_acc = self.compute_val_loss()
            return val_loss, val_acc

    def train(self):
        if not self.args.freeze_meta and not self.args.load_hypernet:
            self.pretrain_confidence_network()
            save_checkpoint({
                    'epoch': "0",
                    'backbone_dict': self.backbone.state_dict(),
                    'classifier_dict': self.classifer_net.state_dict(),
                    'confidence_dict': self.confidence_net.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'meta_optimizer': self.meta_optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'meta_scheduler': self.meta_scheduler.state_dict()
                }, self.experiment_saved_models, filename="pretrained_confdence_network.pth.tar")

        if self.args.pre_train:
            self.pre_train()
            self.meta_update()
            save_checkpoint({
                'epoch': "0",
                'backbone_dict': self.backbone.state_dict(),
                'classifier_dict': self.classifer_net.state_dict(),
                'confidence_dict': self.confidence_net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'meta_optimizer': self.meta_optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'meta_scheduler': self.meta_scheduler.state_dict()
            }, self.experiment_saved_models, filename="confidence_network_update_post_warmup_training.pth.tar")
        
        end = time.time()
        for epoch in range(self.args.hyper_epochs):
            ####################################
            # Re-initialize Average Meters
            ####################################
            self.batch_time = AverageMeter()
            self.data_time = AverageMeter()
            self.losses = AverageMeter()
            self.losses_x = AverageMeter()
            self.losses_u = AverageMeter()
            self.losses_u_weighted = AverageMeter()
            self.mask_probs = AverageMeter()
            ####################################
            # Inner Loop Training
            ####################################
            if self.args.progress:
                p_bar = tqdm(range(self.args.inner_steps))
            for step in range(self.args.inner_steps):
                self.data_time.update(time.time() - end)
                self.train_step()
                self.batch_time.update(time.time() - end)
                end=time.time()
                if self.args.progress:
                        p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}. ".format(
                        epoch=epoch + 1,
                        epochs=self.args.hyper_epochs,
                        batch=step + 1,
                        iter=self.args.inner_steps,
                        lr=self.scheduler.get_last_lr()[0],
                        data=self.data_time.avg,
                        bt=self.batch_time.avg,
                        loss=self.losses.avg,
                        loss_x=self.losses_x.avg,
                        loss_u=self.losses_u.avg,
                        mask=self.mask_probs.avg))
                p_bar.update()
                if self.args.debugging:
                    self.writer.add_scalar('train_epoch'+str(epoch)+'/1.train_loss', self.losses.val, step)
                    self.writer.add_scalar('train_epoch'+str(epoch)+'/2.train_loss_x', self.losses_x.val, step)
                    self.writer.add_scalar('train_epoch'+str(epoch)+'/3.train_loss_u', self.losses_u.val, step)
                    self.writer.add_scalar('train_epoch'+str(epoch)+'/4.train_loss_u_weighted', self.losses_u_weighted.val, step)
                    self.writer.add_scalar('train_epoch'+str(epoch)+'/5.mask', self.mask_probs.val, step)
                    self.writer.add_scalar('train_epoch'+str(epoch)+'/6.MWN_Outputs_Avg', self.confidence_net_weights[-1], step)
                    self.writer.add_scalar('train_epoch'+str(epoch)+'/7.MWN_Outputs_Std', self.confidence_net_stds[-1], step)
            if self.args.progress:
                    p_bar.close()
            ####################################
            # Meta-Update
            ####################################
            if epoch % self.args.meta_update_freq == 0:
                val_loss, val_acc = self.meta_update()
            else:
                with torch.no_grad():
                    val_loss, val_acc = self.compute_val_loss()
            ####################################
            # Summary Writer
            ####################################
            self.writer.add_scalar('train/1.train_loss', self.losses.avg, epoch)
            self.writer.add_scalar('train/2.train_loss_x', self.losses_x.avg, epoch)
            self.writer.add_scalar('train/3.train_loss_u', self.losses_u.avg, epoch)
            self.writer.add_scalar('train/4.train_loss_u_weighted', self.losses_u_weighted.avg, epoch)
            self.writer.add_scalar('train/4.mask', self.mask_probs.avg, epoch)
            self.writer.add_scalar('test/1.val_loss', val_loss, epoch)
            self.writer.add_scalar('test/2.val_accuracy', val_acc, epoch)
            
            # Save Checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'backbone_dict': self.backbone.state_dict(),
                'classifier_dict': self.classifer_net.state_dict(),
                'confidence_dict': self.confidence_net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'meta_optimizer': self.meta_optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'meta_scheduler': self.meta_scheduler.state_dict()
            }, self.experiment_saved_models, filename="epoch_"+str(epoch+1)+".pth.tar")
        
        ##################################
        # Test
        ##################################
        test_loss, top1_test_acc, top5_test_acc = self.test()
        print("test top-1 acc: {:.2f}".format(top1_test_acc))
        print("test top-5 acc: {:.2f}".format(top5_test_acc))
        print("test loss: {:.2f}".format(test_loss))
        self.writer.add_scalar('test/3.test_loss', test_loss, epoch)
        self.writer.add_scalar('test/4.test_top1_accuracy', top1_test_acc, epoch)
        self.writer.add_scalar('test/5.test_top5_accuracy', top5_test_acc, epoch)