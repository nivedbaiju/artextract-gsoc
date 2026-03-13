import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from dataset import WikiArtSupervisedDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

def getdataloader(csv_path, root_dir):

    # augmentations performed: EfficientNet b3(my CNN backbone) uses 300x300 inputs so we do RandomResizedCrop(300), then images are randomly flipped horizontally with a probability of 0.5
    # we only apply mild colorjitter to preserve the original colours and textures of the painting
    # the image is converted to a tensor and normalised using mean and standard deviation of Imagenet.
    train_transform=v2.Compose([v2.RandomResizedCrop(300),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.02),
                v2.ToTensor(),
                v2.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    val_transform=v2.Compose([v2.Resize(300),v2.CenterCrop(300),
                              v2.ToTensor(),
                                v2.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])
    dataset= WikiArtSupervisedDataset(csv_path=csv_path, root_dir=root_dir,transform=None)
    #see no of images
    print(len(dataset))

    labels=dataset.artist_labels
    #here i am using stratified split to avoid class imbalance
    train_indices,val_indices= train_test_split(range(len(dataset)), test_size=0.2, stratify=labels, random_state=42)
    
    train_dataset= WikiArtSupervisedDataset(csv_path=csv_path, root_dir=root_dir,transform=train_transform)
    val_dataset= WikiArtSupervisedDataset(csv_path=csv_path, root_dir=root_dir,transform=val_transform)
    
    train_dataset= Subset(train_dataset, train_indices)
    val_dataset= Subset(val_dataset, val_indices)

    train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True,num_workers=4,pin_memory=True)
    val_loader=DataLoader(val_dataset,batch_size=32,shuffle=False,num_workers=4,pin_memory=True)
    return train_loader, val_loader

def compute_weight(labels):
    labels=torch.tensor(labels)
    class_counts = torch.bincount(labels)
    num_classes = len(class_counts)
    total_samples = len(labels)

    weights=total_samples / (num_classes * class_counts.clamp(min=1).float())
    return weights
