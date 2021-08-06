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
    model = build_wideresnet(depth=args.model_depth,
                             widen_factor=args.model_width,
                             dropout=0,
                             num_classes=10)

    meta_model = build_wideresnet(depth=args.model_depth,
                             widen_factor=args.model_width,
                             dropout=0,
                             num_classes=1) # Output is Importance of Unlabelled Image512

    ######################################
    # Create Optimizers
    ######################################
    # Create Classification Network Optimizer
    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    total_steps = (args.hyper_epochs*args.inner_steps)+(args.pre_train_epochs*args.pre_train_steps)

    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, total_steps)

    # Create Confidence Network Optimizer
    no_decay = ['bias', 'bn']
    meta_grouped_parameters = [
        {'params': [p for n, p in meta_model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in meta_model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    meta_optimizer = optim.SGD(meta_grouped_parameters, lr=args.hyper_lr,
                                momentum=0.9, nesterov=args.nesterov)
    meta_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(meta_optimizer, args.hyper_epochs + args.pre_train)


    model.zero_grad()
    meta_model.zero_grad()

    #############################################
    # Load Confidence Network Checkpoint
    #############################################
    if args.load_hypernet:
        checkpoint = torch.load(args.checkpoint)
        meta_model.load_state_dict(checkpoint['meta_model_dict'])

    ######################################
    # Start Training
    ######################################
    experiment  = ExperimentBuilder(model, meta_model, optimizer, scheduler, meta_optimizer, meta_scheduler, args, args.experiment_name, data_loader)
    experiment.train()
    
if __name__ == '__main__':
    main()