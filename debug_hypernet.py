import torch
import argparse
from Models.WideResnet import build_wideresnet
import os
from DatasetsUtil.DataLoaderWrap import DataLoaderWrap
from torch.utils.data import  RandomSampler
import csv
from tqdm import tqdm

def save_statistics(experiment_log_dir, filename, stats_dict, current_epoch, continue_from_mode = False, save_full_dict=False):
    """
    Saves the statistics in stats dict into a csv file. Using the keys as the header entries and the values as the
    columns of a particular header entry
    :param experiment_log_dir: the log folder dir filepath
    :param filename: the name of the csv file
    :param stats_dict: the stats dict containing the data to be saved
    :param current_epoch: the number of epochs since commencement of the current training session (i.e. if the experiment continued from 100 and this is epoch 105, then pass relative distance of 5.)
    :param save_full_dict: whether to save the full dict as is overriding any previous entries (might be useful if we want to overwrite a file)
    :return: The filepath to the summary file
    """
    summary_filename = os.path.join(experiment_log_dir, filename)
    mode = 'a' if continue_from_mode else 'w'
    with open(summary_filename, mode) as f:
        writer = csv.writer(f)
        if not continue_from_mode:
            writer.writerow(list(stats_dict.keys()))

        if save_full_dict:
            total_rows = len(list(stats_dict.values())[0])
            for idx in range(total_rows):
                row_to_add = [value[idx] for value in list(stats_dict.values())]
                writer.writerow(row_to_add)
        else:
            row_to_add = [value[current_epoch] for value in list(stats_dict.values())]
            writer.writerow(row_to_add)

    return summary_filename

def main(args, data_loader):
    meta_model = build_wideresnet(depth=28,
                             widen_factor=2,
                             dropout=0,
                             num_classes=1)

    saved_models = os.path.abspath(os.path.join(args.dir, "saved_models"))
    checkpoint = torch.load(os.path.join(saved_models, args.checkpoint_name))

    meta_model.load_state_dict(checkpoint['meta_model_dict'])

    targets = []
    with torch.no_grad():
        for i in range (5):
            x, x_u, x_w = data_loader.get_unlabeled_batch()
            targets += meta_model(x_u)
            
    return torch.mean(torch.stack(targets)), torch.std(torch.stack(targets))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--dir', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--checkpoint_name', default='', type=str, help='checkpoint to load')
    parser.add_argument('--data', type=str, default='CIFAR10', choices=['CIFAR10', 'SVHN'], help='Name of Dataset (CIFAR10, MNIST or SVHN)')
    parser.add_argument('--train_labelled_size', type=int, default=250, metavar='N', help='Number of training labelled samples from the dataset')
    parser.add_argument('--validation_size', type=int, default=1000, metavar='N', help='Number of validation labelled samples from the dataset')
    parser.add_argument('--labelled_batch_size', type=int, default=64, metavar='N', help='Batch size for each inner loop epoch')
    parser.add_argument('--mu', type=int, default=7, metavar='N', help='Coefficient of unlabeled batch size')
    parser.add_argument('--start_epoch', type=int, default=1, metavar='N', help='Epoch checkpoint to start evaluating at')
    parser.add_argument('--end_epoch', type=int, default=100, metavar='N', help='Final Epoch checkpoint to stop evaluating at')
    parser.add_argument('--warmup', type=bool, default=False, metavar='N', help='If True, includes initial confidence network vals after warm-up and pre-training')


    args = parser.parse_args()

    data_loader = DataLoaderWrap(args, RandomSampler, 0, 0)

    progress = tqdm(range(args.start_epoch, args.end_epoch))
    results = {'means':[],'stds':[]}

    if args.warmup:
        print("Including Warmup and Pre-Trained Update of Confidence Network")
        args.checkpoint_name= "confidence_network_update_post_warmup_training.pth.tar"
        mean, std = main(args, data_loader)
        results['means'].append(mean.item())
        results['stds'].append(std.item())

        args.checkpoint_name= "pretrained_confdence_network.pth.tar"
        mean, std = main(args, data_loader)
        results['means'].append(mean.item())
        results['stds'].append(std.item())

    for i in range(args.start_epoch, args.end_epoch):
        args.checkpoint_name= "epoch_"+str(i)+".pth.tar"
        mean, std = main(args, data_loader)
        results['means'].append(mean.item())
        results['stds'].append(std.item())
        progress.update()
        
    save_statistics(args.dir, "MWN_Outputs.csv", results, 0, False, True)