import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from dataset import WikiArtSupervisedDataset

def getdataloader(train_csv,val_csv, root_dir,artist_map,genre_map,style_map):

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


    train_dataset = WikiArtSupervisedDataset(
        root_dir,
        train_csv,
        artist_map,
        genre_map,
        style_map,
        transform=train_transform
    )

    val_dataset = WikiArtSupervisedDataset(
        root_dir,
        val_csv,
        artist_map,
        genre_map,
        style_map,
        transform=val_transform
    )

    print(f"No of training samples:",len(train_dataset))
    print(f"No of validation samples:",len(val_dataset))

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

def weight_values(csv_path, root_dir, artist_map, genre_map, style_map):
    dataset= WikiArtSupervisedDataset(csv_path, root_dir,artist_map,genre_map,style_map,transform=None)
    style_weights=compute_weight(dataset.style_labels)
    genre_weights=compute_weight(dataset.genre_labels)
    artist_weights=compute_weight(dataset.artist_labels)
    return style_weights, genre_weights, artist_weights