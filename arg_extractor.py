import argparse
import time

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser(
        description='Welcome to the MLP course\'s Pytorch training and inference helper script')

    # General Arguments
    parser.add_argument('--experiment_name', nargs="?", type=str, default="exp_1", help='Experiment Name - to be used for building the experiment folder')
    parser.add_argument('--data', type=str, default='CIFAR10', choices=['CIFAR10', 'SVHN'], help='Name of Dataset (CIFAR10, MNIST or SVHN)')
    parser.add_argument('--model_depth', type=int, default=28, help='Wide Resnet Model Depth (default WideResnet-28-2)')
    parser.add_argument('--model_width', type=int, default=2, help='Wide Resnet Model Width (default WideResnet-28-2)')
    parser.add_argument('--hyper_epochs', type=int, default=1000, help='meta-optimization upper epoch limit')
    parser.add_argument('--inner_steps', type=int, default=512, help='inner loop per epoch upper steps limit')
    parser.add_argument('--train_labelled_size', type=int, default=250, metavar='N', help='Number of training labelled samples from the dataset')
    parser.add_argument('--validation_size', type=int, default=1000, metavar='N', help='Number of validation labelled samples from the dataset')
    parser.add_argument('--labelled_batch_size', type=int, default=64, metavar='N', help='Batch size for each inner loop epoch')
    parser.add_argument('--mu', type=int, default=7, metavar='N', help='Coefficient of unlabeled batch size')
    parser.add_argument('--meta_update_freq', type=int, default=1, metavar='N', help='How many inner epochs before each meta-update')
    parser.add_argument('--num_neumann_terms', type=int, default=3, help='The maximum number of neumann terms to use')
    parser.add_argument('--pre_train', type=bool, default=True, help='Pre-Train Model before IFT Meta-Learning')
    parser.add_argument('--pre_train_epochs', type=int, default=10, help='Number of epochs to Pre-Train Model for before IFT Meta-Learning')
    parser.add_argument('--pre_train_steps', type=int, default=512, help='Number of steps per epoch to Pre-Train Model for before IFT Meta-Learning')
    parser.add_argument('--T', default=1, type=float, help='pseudo label temperature')
    parser.add_argument('--wdecay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float, help='Initial Classifier Network Learning Rate')
    parser.add_argument('--hyper_lr', '--hyper_learning-rate', default=1e-3, type=float, help='Confidence Network Learning Rate')
    parser.add_argument('--nesterov', action='store_true', default=True, help='use nesterov momentum')
    parser.add_argument('--warmup', default=0, type=float, help='warmup epochs (unlabeled data based)')
    parser.add_argument('--threshold', default=0.95, type=float, help='pseudo label threshold')
    parser.add_argument('--progress', action='store_true', default=True, help='Show Progress Bar')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--use_gpu', action='store_true', default=True, help='Flag to use GPU/CUDA (ENABLED by default)')
    parser.add_argument('--debugging', action='store_true', default=True, help='If Debugging, Plot Every Epoch on Tensoboard')
    parser.add_argument('--approx', type=str, default='identity', choices=['numn', 'identity'], help='Inverse Hessian Approximation to Use (numn or identity)')
    parser.add_argument('--freeze_meta', type=bool, default=False, help='If True, The Confidence Network is fixed/Freezed (no meta-updates are executed).')
    parser.add_argument('--checkpoint', default='', type=str, help='File path of the checkpoint to load.')
    parser.add_argument('--load_hypernet', type=bool, default=False, help='If true, the checkpoint is used to load the hypernet (MLP weighting head - confidence network).')
    parser.add_argument('--load_backbone', type=bool, default=False, help='If true, the checkpoint is used to load the backbone (feature extractor).')
    parser.add_argument('--load_classifier', type=bool, default=False, help='If true, the checkpoint is used to load the classifier head (MLP classifier).')

                    
    args = parser.parse_args()
    print(args)
    return args