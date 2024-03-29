# This script is adapted from: https://github.com/kekmodel/FixMatch-pytorch/blob/f54946074fba383e28320d8f50b627eabd0c7e3c/dataset/cifar.py

import math
import numpy as np
from torchvision import datasets
from torchvision import transforms
from .randaugment import RandAugmentMC
from DatasetsUtil.TransformFixMatch import TransformFixMatch
from DatasetsUtil.CIFAR10SSL import CIFAR10SSL
from DatasetsUtil.SVHN_SSL import SVHN_SSL

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
svhn_mean = (0.4377, 0.4438, 0.4728)
svhn_std = (0.0392, 0.0404, 0.0388)

def get_cifar10(args, root="data"):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs, validation_idx = x_u_split(
        args, base_dataset.targets, 10)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

    validation_labeled_dataset = CIFAR10SSL(
        root, validation_idx, train=True,
        transform=transform_labeled)
    
    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, validation_labeled_dataset, test_dataset

def x_u_split(args, labels, num_classes):
    label_per_class_train = args.train_labelled_size // num_classes
    label_per_class_val = args.validation_size // num_classes
    total_train_val_labelled_ = label_per_class_train + args.validation_size
    labels = np.array(labels)
    labeled_idx = []
    val_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    # Refer to Note 2 at the bottom of page 3: https://arxiv.org/pdf/2001.07685.pdf
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, total_train_val_labelled_, False)
        labeled_idx.extend(idx[0:label_per_class_train])
        val_idx.extend(idx[label_per_class_train:label_per_class_val+label_per_class_train])
    labeled_idx = np.array(labeled_idx)
    val_idx = np.array(val_idx)
    assert len(labeled_idx) == args.train_labelled_size
    assert len(val_idx) == args.validation_size

    if args.train_labelled_size < args.labelled_batch_size:
        num_expand_x = math.ceil(
            args.labelled_batch_size * args.inner_steps / args.train_labelled_size)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)

    unlabeled_idx = unlabeled_idx[~np.in1d(unlabeled_idx, val_idx)]

    assert len(np.unique(np.unique(np.concatenate([unlabeled_idx, val_idx])))) == len(labels)

    return labeled_idx, unlabeled_idx, val_idx

def get_svhn(args, root="data"):
    transform_labeled = transforms.compose([
        #transforms.RandomHorizontalFlip(), according to sec 2.3, flip is not applied to svhn: https://arxiv.org/pdf/2001.07685.pdf
        transforms.RandomCrop(size=32,
        padding=int(32*0.125),
        padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=svhn_mean, std=svhn_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=svhn_mean, std=svhn_std)
    ])
    base_dataset = datasets.SVHN(root, split="train", download=True)

    train_labeled_idxs, train_unlabeled_idxs, validation_idx = x_u_split(args, base_dataset.targets, 10)

    train_labeled_dataset = SVHN_SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = SVHN_SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

    validation_labeled_dataset = SVHN_SSL(
        root, validation_idx, train=True,
        transform=transform_labeled)

    test_dataset = datasets.SVHN(
        root, split="test", transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, validation_labeled_dataset, test_dataset

DATASET_GETTERS = {'CIFAR10': get_cifar10,
                   'SVHN': get_svhn}