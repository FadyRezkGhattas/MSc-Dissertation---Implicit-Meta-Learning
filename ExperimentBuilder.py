import torch
import torch.nn.functional as F
import torch.nn as nn
from DatasetsUtil.DataLoaderWrap import DataLoaderWrap
from util.experiment_util import AverageMeter, interleave, de_interleave, accuracy
import time
import os
from tqdm import tqdm

class ExperimentBuilder(nn.Module):
    def __init__(self, network_model, meta_model, optimizer, scheduler, meta_optimizer, meta_scheduler, args, experiment_name, data_loader : DataLoaderWrap) -> None:
        super(ExperimentBuilder, self).__init__()
        self.experiment_name = experiment_name
        self.model = network_model
        self.data_loader = data_loader
        self.args = args
        self.meta_model = meta_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.meta_optimizer = meta_optimizer
        self.meta_scheduler = meta_scheduler

        if torch.cuda.device_count() > 1 and args.use_gpu:
            self.device = torch.cuda.current_device()
            self.model.to(self.device)
            self.meta_model.to(self.device)
            self.model = nn.DataParallel(module=self.model)
            self.meta_model = nn.DataParallel(module=self.meta_model)
            print('Use Multi GPU', self.device)
        elif torch.cuda.device_count() == 1 and args.use_gpu:
            self.device = torch.cuda.current_device()
            print("device is ", self.device)
            # sends the model from the cpu to the gpu
            self.model.to(self.device)
            self.meta_model.to(self.device)
            print('Use GPU', self.device)
        else:
            print("use CPU")
            self.device = torch.device('cpu')  # sets the device to be CPU
            print(self.device)

        self.experiment_folder = os.path.abspath(self.experiment_name)
        self.experiment_logs = os.path.abspath(
            os.path.join(self.experiment_folder, "result_outputs"))
        self.experiment_saved_models = os.path.abspath(
            os.path.join(self.experiment_folder, "saved_models"))
        self.experiment_plots = os.path.abspath(
            os.path.join(self.experiment_folder, "plots"))
        # If experiment directory does not exist
        if not os.path.exists(self.experiment_folder):
            os.mkdir(self.experiment_folder)  # create the experiment directory
            # create the experiment log directory
            os.mkdir(self.experiment_logs)
            # create the experiment saved models directory
            os.mkdir(self.experiment_saved_models)
            os.mkdir(self.experiment_plots)

        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.losses = AverageMeter()
        self.losses_x = AverageMeter()
        self.losses_u = AverageMeter()
        self.mask_probs = AverageMeter()

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

        grad = torch.autograd.grad(f, w(), grad_outputs=v, retain_graph=True)  # (L^2_t/dwdw)*(dv/dw)
        for j in range(i):
            v = [v_ - alpha * g for v_, g in zip(v, grad)]
            # p += v (Typo in the arxiv version of the paper)
            p = [p_ + v_ for p_, v_ in zip(p, v)]

        return p

    def hypergradient(self, validation_loss, training_loss, lambda_, w):
        v1 = torch.autograd.grad(validation_loss, w(), retain_graph=True)
        d_train_d_w = torch.autograd.grad(training_loss, w(), create_graph=True)
        v2 = self.neummann_approximation(v1, d_train_d_w, w, i=self.args.num_neumann_terms)
        v3 = torch.autograd.grad(d_train_d_w, lambda_(), grad_outputs=v2, retain_graph=True)
        # d_val_d_lambda = torch.autograd.grad(validation_loss, lambda_())
        #return [d - v for d, v in zip(d_val_d_lambda, v3)]
        return [-1*v for v in v3]

    def compute_batch_loss(self):
        self.model.train()
        inputs_x, targets_x = self.data_loader.get_labelled_batch()
        inputs_x, targets_x = inputs_x.to(self.device), targets_x.to(self.device)

        batch_size = inputs_x.shape[0]

        inputs_u_w, inputs_u_s = self.data_loader.get_unlabeled_batch()
        inputs_u_w, inputs_u_s = inputs_u_w.to(self.device), inputs_u_s.to(self.device)
        inputs = interleave(
            torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*self.args.mu+1)

        logits = self.model(inputs)
        logits = de_interleave(logits, 2*self.args.mu+1)
        logits_x = logits[:batch_size]
        logits_u_w, logits_u_s = logits[batch_size:].chunk(2)

        del logits

        Lx = torch.nn.CrossEntropyLoss()(logits_x, targets_x)

        pseudo_label = torch.softmax(logits_u_w.detach()/self.args.T, dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(self.args.threshold).float().to(self.device)

        Lu = F.cross_entropy(logits_u_s, targets_u, reduction='none')
        
        u_weight = self.meta_model(inputs_u_w)

        weighted_lu =  u_weight[:,0] * Lu
        weighted_lu = weighted_lu.mean()

        loss = Lx + weighted_lu

        return loss, Lx, weighted_lu, mask

    def compute_val_loss(self):
        self.model.train()
        inputs, targets = self.data_loader.val_data_x.to(self.device), self.data_loader.val_data_y.to(self.device)
        logits = self.model(inputs)
        return torch.nn.CrossEntropyLoss()(logits, targets)

    def train_step(self, pre_train = False):
        loss, Lx, weighted_lu, mask = self.compute_batch_loss()

        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        self.losses.update(loss.item())
        self.losses_x.update(Lx.item())
        self.losses_u.update(weighted_lu.item())
        self.mask_probs.update(mask.mean().item())
    
    def pre_train(self):
        end = time.time()
        self.model.train()

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
                if self.args.progress:
                    p_bar.close()
        print("########################################################################")
        print("Pre-Training Finished. Starting Meta-Learning Routine.")
        print("########################################################################")

    def test(self):
        with torch.no_grad():
            self.model.eval()
            inputs = self.data_loader.test_data_x
            targets = self.data_loader.test_data_y
            outputs = self.model(inputs)
            loss = torch.nn.CrossEntropyLoss()(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1,5))
            return loss, prec1, prec5

    def meta_update(self):
        train_loss, Lx, weighted_lu, mask = self.compute_batch_loss()
        val_loss = self.compute_val_loss()
        
        hyper_grads = self.hypergradient(val_loss, train_loss, self.meta_model.parameters, self.model.parameters)

        self.meta_optimizer.zero_grad()
        for p, g in zip(self.meta_model.parameters(), hyper_grads):
            p.grad = g
        self.meta_optimizer.step()
        self.meta_scheduler.step()

    def train(self):
        if self.args.pre_train:
            self.pre_train()
            self.meta_update()
        
        end = time.time()
        for epoch in range(self.args.hyper_epochs):
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
            if self.args.progress:
                    p_bar.close()
            ####################################
            # Meta-Update
            ####################################
            self.meta_update()