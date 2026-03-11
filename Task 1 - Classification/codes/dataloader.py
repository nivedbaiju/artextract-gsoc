import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
import sys
from torchvision.transforms import v2
from preprocessing import WikiArtSupervisedDataset

def getdataloader(csv_path, root_dir):

    # augmentations performed: EfficientNet b3(my CNN backbone) uses 300x300 inputs so we do RandomResizedCrop(300), then images are randomly flipped horizontally with a probability of 0.5
    # we only apply mild colorjitter to preserve the original colours and textures of the painting
    # the image is converted to a tensor and normalised using mean and standard deviation of Imagenet.
    transform=v2.Compose([v2.RandomResizedCrop(300),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.02),
                v2.ToTensor(),
                v2.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    dataset= WikiArtSupervisedDataset(csv_path=csv_path, root_dir=root_dir,transform=transform)
    #see no of images
    print(len(dataset))

    dataloader=DataLoader(dataset,batch_size=32,shuffle=True,num_workers=4)
    return dataloader
