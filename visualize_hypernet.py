import torch
import argparse
from Models.WideResnet import build_wideresnet
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
plt.style.use("seaborn-darkgrid")
plt.rcParams.update({"font.size": 10})
import seaborn as sns

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

def argmedian(x):
  return np.argpartition(x, len(x) // 2)[len(x) // 2]

def main(args):
    device = torch.cuda.current_device()
    meta_model = build_wideresnet(depth=28,
                             widen_factor=2,
                             dropout=0,
                             num_classes=1)

    training_data = datasets.CIFAR10(root="data",
                                            train=True,
                                            download=True,
                                            transform=transforms.Compose([
                                            transforms.ToTensor()
                                            ]))
    data_loader = DataLoader(training_data, batch_size=args.labelled_batch_size, shuffle=True)
    data_loader = iter(data_loader)

    saved_models = os.path.abspath(os.path.join(args.dir, "saved_models"))
    checkpoint = torch.load(os.path.join(saved_models, args.checkpoint_name))

    meta_model.load_state_dict(checkpoint['meta_model_dict'])
    meta_model = meta_model.to(device)

    x_weights = []
    images = []
    targets = []
    n_batches = round(args.train_labelled_size/args.labelled_batch_size)
    progress = tqdm(range(0, n_batches))
    with torch.no_grad():
        for i in range (n_batches):
            inputs, outputs = next(data_loader)
            images = inputs.permute(0, 2, 3, 1).numpy() if i == 0 else np.concatenate((images, inputs.permute(0, 2, 3, 1).numpy()))
            targets = outputs.numpy() if i == 0 else np.concatenate((targets, outputs.numpy()))

            inputs = inputs.to(device)
            normalize = transforms.Normalize(mean=cifar10_mean, std=cifar10_std)

            weights  = meta_model(normalize(inputs))
            x_weights = weights.cpu().numpy().ravel() if i == 0 else np.concatenate([x_weights, weights.cpu().numpy().ravel()])
            progress.update()
    
    cifar_keys = {
        '0': 	'airplane',
        '1': 	'automobile',
        '2': 	'bird',
        '3': 	'cat',
        '4': 	'deer',
        '5': 	'dog',
        '6': 	'frog',
        '7': 	'horse',
        '8': 	'ship',
        '9': 	'truck'
    }

    
    for i in range(10):
        class_idx = np.where(targets == i)[0]
        idx = np.argmax(x_weights[class_idx])
        img = images[class_idx][idx]
        img_name = "CN_Images/"+cifar_keys[str(i)]+"_BestImageWeight_"+format(x_weights[class_idx][idx], '.2f')+".png"
        plt.imsave(img_name, img)

        idx = argmedian(x_weights[class_idx])
        img = images[class_idx][idx]
        img_name = "CN_Images/"+cifar_keys[str(i)]+"_MedianImageWeight_"+format(x_weights[class_idx][idx], '.2f')+".png"
        plt.imsave(img_name, img)
        
        idx = np.argmin(x_weights[class_idx])
        img = images[class_idx][idx]
        img_name = "CN_Images/"+cifar_keys[str(i)]+"_WorstImageWeight_"+format(x_weights[class_idx][idx], '.2f')+".png"
        plt.imsave(img_name, img)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--dir', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--checkpoint_name', default='', type=str, help='checkpoint to load')
    parser.add_argument('--train_labelled_size', type=int, default=50000, metavar='N', help='Number of training labelled samples from the dataset')
    parser.add_argument('--labelled_batch_size', type=int, default=500, metavar='N', help='Batch size for each inner loop epoch')
    parser.add_argument('--warmup', type=bool, default=False, metavar='N', help='If True, includes initial confidence network vals after warm-up and pre-training')


    args = parser.parse_args()

    args.dir = "/home/fady/Desktop/edinburgh/Dissertation/Phase2_results/mwns/mwn_cifar10@250_1"
    args.checkpoint_name= "best_val_loss.pth.tar"
    x_weights = main(args)
        
    #save_statistics(args.dir, "MWN_Outputs.csv", results, 0, False, True)