
import os
import argparse
import json

import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms

import deeplabv3
import mfpanet

from utils import replace_batchnorm2d, smart_group_norm
from data import SegmentationDatasetWrapper, LabelmeDataset, BinarySegmentationDataset
from losses import FocalTverskyLoss
from trainer import SegmentationTrainer

### parse arguments ###
parser = argparse.ArgumentParser()

parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight-decay', type=float, default=5e-5)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--img-dir', type=str, help='Path of image directory')
parser.add_argument('--mask-dir', type=str, help='Path of mask directory')
parser.add_argument('--val-img-dir', type=str, help='Path of validation image directory')
parser.add_argument('--val-mask-dir', type=str, help='Path of validation mask directory')
parser.add_argument('--img-size', type=int, default=256, help='Size of images.')
parser.add_argument('--strong-aug', type=bool, default=False, help='Whether to perform strong augmentation.')
parser.add_argument('--model', type=str, default='MFPAN', help='Name of model.')
parser.add_argument('--pretrained', default=False, help='Whether to use the pretrained model.')
parser.add_argument('--norm-layer', type=str, default='gn', help='Available options are ["bn", "gn", "in"], represent batch, group and instance normalization, respectively.')
parser.add_argument('--workers', type=int, default=0)
parser.add_argument('--use-amp', type=bool, default=True)
parser.add_argument('--use-cuda', type=bool, default=True)
parser.add_argument('--save-path', type=str, default=None, help='The path of the trained model will be saved.')
parser.add_argument('--save-history', type=str, default=None, help='The path of the learning curve will be saved.')


args = parser.parse_args()

def get_model(name, classes, pretrained=False):
    if hasattr(deeplabv3, name):
        return getattr(deeplabv3, name)(classes, pretrained_backbone=pretrained)
    elif hasattr(mfpanet, name):
        return getattr(mfpanet, name)(classes, pretrained=pretrained)
    else:
        raise ValueError(f'Invalid model: {name}.')


def main():
    ### prepare dataset ###
    normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if args.strong_aug:
        spatial_transform = transforms.Compose([
            transforms.RandomResizedCrop(args.img_size),
            transforms.RandomErasing(0.2),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])
        color_transform = transforms.Compose([
            transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
            normalize_transform
        ])
    else:
        spatial_transform = transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])
        color_transform = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2),
            normalize_transform
        ])

    train_ds = BinarySegmentationDataset(
        args.img_dir, args.mask_dir,
        transforms.ToTensor(), transforms.ToTensor())
    train_ds = SegmentationDatasetWrapper(train_ds,
                                          spatial_transform=spatial_transform,
                                          color_transform=color_transform)


    if args.val_img_dir is not None and args.val_mask_dir is not None:
        test_ds = BinarySegmentationDataset(
            args.val_img_dir, args.val_mask_dir,
            transforms.ToTensor(), transforms.ToTensor())
        test_ds = SegmentationDatasetWrapper(test_ds,
                                             spatial_transform=transforms.Resize(args.img_size),
                                             color_transform=normalize_transform)

    train_loader = torch.utils.data.DataLoader(
            train_ds,
            sampler=torch.utils.data.RandomSampler(train_ds, replacement=True),
            batch_size=args.batch_size,
            num_workers=args.workers,
            drop_last=True)
    test_loader = torch.utils.data.DataLoader(
            test_ds,
            sampler=torch.utils.data.SequentialSampler(test_ds),
            batch_size=args.batch_size,
            num_workers=args.workers,
            drop_last=True)

    ### build model ###
    CLASSES = 2
    model = get_model(args.model, CLASSES, pretrained=args.pretrained)

    if args.norm_layer == 'gn':
        replace_batchnorm2d(model, smart_group_norm)
    elif args.norm_layer == 'in':
        replace_batchnorm2d(model, lambda n: nn.InstanceNorm2d(n, affine=True))

    ### training ###
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs*len(train_loader), args.lr * 1e-2)
    criterion = FocalTverskyLoss(activation='softmax')

    trainer = SegmentationTrainer(model, optimizer, criterion,
                                  lr_scheduler=lr_scheduler, use_amp=True, use_cuda=True)

    history = trainer.fit(train_loader, val_loader=test_loader, epochs=args.epochs)

    ### save results ###
    if args.save_path is not None:
        torch.save(model.state_dict(), args.save_path)

    if args.save_history is not None:
        with open(args.save_history, "w") as f:
            json.dump(history, f)

if __name__ == "__main__":
    main()