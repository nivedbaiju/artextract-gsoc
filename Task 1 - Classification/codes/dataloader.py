import torch
from torch.utils.data import Dataset, DataLoader,WeightedRandomSampler
from torchvision.transforms import v2
from dataset import WikiArtSupervisedDataset
root_dir=r"C:\Users\nived\artextract-gsoc\Task 1 - Classification\datasets\wikiart_filtered"
train_csv=r"C:\Users\nived\Downloads\wikiart_csv\train_labels_fixed.csv"
val_csv=r"C:\Users\nived\Downloads\wikiart_csv\val_labels_fixed.csv"
artist_map=r"C:\Users\nived\Downloads\wikiart_csv\artist_class.txt"
genre_map=r"C:\Users\nived\Downloads\wikiart_csv\genre_class.txt"
style_map=r"C:\Users\nived\Downloads\wikiart_csv\style_class.txt"

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

    genre_labels = torch.tensor(train_dataset.genre_labels)
    style_labels = torch.tensor(train_dataset.style_labels)

    style_counts = torch.bincount(torch.tensor(style_labels))
    genre_counts = torch.bincount(torch.tensor(genre_labels))
    
    style_weights=1.0/style_counts.clamp(min=1).float()
    genre_weights=1.0/genre_counts.clamp(min=1).float()

    sample_weights = style_weights[style_labels]+ genre_weights[genre_labels]

    sampler=WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader=DataLoader(train_dataset,batch_size=8,sampler=sampler,num_workers=2,pin_memory=True,persistent_workers=True)
    val_loader=DataLoader(val_dataset,batch_size=8,shuffle=False,num_workers=2,pin_memory=True,persistent_workers=True)
    return train_loader, val_loader

def compute_weight(labels):
    labels=torch.tensor(labels)
    class_counts = torch.bincount(labels)
    num_classes = len(class_counts)
    total_samples = len(labels)

    weights=total_samples / (num_classes * class_counts.clamp(min=1).float())
    return weights

def weight_values(csv_path, root_dir, artist_map, genre_map, style_map):
    dataset= WikiArtSupervisedDataset(root_dir,csv_path,artist_map,genre_map,style_map,transform=None)
    style_weights=compute_weight(dataset.style_labels)
    genre_weights=compute_weight(dataset.genre_labels)
    artist_weights=compute_weight(dataset.artist_labels)
    return style_weights, genre_weights, artist_weights