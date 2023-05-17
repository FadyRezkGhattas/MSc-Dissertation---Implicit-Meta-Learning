import numpy as np
import torch
from arg_extractor import get_args
from Models.WideResnet import build_wideresnet
from Models.MLP import MLP
import os
from torch.utils.data import  RandomSampler
from DatasetsUtil.DataLoaderWrap import DataLoaderWrap
import torch.optim as optim
from util.experiment_util import get_cosine_schedule_with_warmup
from ExperimentBuilder import ExperimentBuilder

def main():
    # Initialization
    args = get_args()  # get arguments from command line
    rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment
    torch.manual_seed(seed=args.seed)  # sets pytorch's seed
    
    ######################################
    # Load Dataset
    ######################################
    data_loader = DataLoaderWrap(args, RandomSampler, 0, 0)

    ######################################
    # Create Models
    ######################################
    backbone = build_wideresnet(depth=args.model_depth,
                             widen_factor=args.model_width,
                             dropout=0,
                             feature_extractor=True)
    classifier_net = MLP(input_dim=backbone.channels, nhid=int(backbone.channels/2), output_dim=10)
    confidence_net = MLP(input_dim=backbone.channels, nhid=int(backbone.channels/2), output_dim=1)

    ######################################
    # Create Optimizers
    ######################################
    # Create backbone and classifier optimizer
    no_decay = ['bias', 'bn']
    backbone_grouped_parameters = [
        {
            'params': [p for n, p in backbone.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.wdecay
        },
        {
            'params': [p for n, p in backbone.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    classifier_grouped_parameters = [
        {'params': [p for n, p in classifier_net.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in classifier_net.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    for i in range(len(classifier_grouped_parameters)):
        backbone_grouped_parameters[i]['params'] += classifier_grouped_parameters[i]['params']
        assert(backbone_grouped_parameters[i]['weight_decay'] == classifier_grouped_parameters[i]['weight_decay'])

    backbone_classifier_optimizer = optim.SGD(backbone_grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    total_steps = (args.hyper_epochs*args.inner_steps)+(args.pre_train_epochs*args.pre_train_steps*args.pre_train)

    backbone_classifier_scheduler = get_cosine_schedule_with_warmup(backbone_classifier_optimizer, args.warmup, total_steps)

    # Create confidence network optimizer
    no_decay = ['bias', 'bn']
    meta_grouped_parameters = [
        {'params': [p for n, p in confidence_net.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in confidence_net.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    meta_optimizer = optim.SGD(meta_grouped_parameters, lr=args.hyper_lr,
                                momentum=0.9, nesterov=args.nesterov)
    meta_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(meta_optimizer, args.hyper_epochs + args.pre_train)


    backbone.zero_grad()
    classifier_net.zero_grad()
    confidence_net.zero_grad()

    #############################################
    # Load Network Checkpoints
    #############################################
    if args.load_hypernet:
        checkpoint = torch.load(args.checkpoint)
        confidence_net.load_state_dict(checkpoint['confidence_dict'])
        print("Meta Model Loaded")
    if args.load_backbone:
        checkpoint = torch.load(args.checkpoint)
        backbone.load_state_dict(checkpoint['backbone_dict'])
        print("Base Model Loaded")
    if args.load_classifier:
        checkpoint = torch.load(args.checkpoint)
        classifier_net.load_state_dict(checkpoint['classifier_dict'])
        print("Base Model Loaded")
    ######################################
    # Start Training
    ######################################
    experiment  = ExperimentBuilder(backbone, classifier_net, confidence_net, backbone_classifier_optimizer, backbone_classifier_scheduler, meta_optimizer, meta_scheduler, args, args.experiment_name, data_loader)
    experiment.train()
    
if __name__ == '__main__':
    main()