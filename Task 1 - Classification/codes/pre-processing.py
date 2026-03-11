import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

# augmentations performed: EfficientNet b3(my CNN backbone) uses 300x300 inputs so we do RandomResizedCrop(300), then images are randomly flipped horizontally with a probability of 0.5
# we only apply mild colorjitter to preserve the original colours and textures of the painting
# the image is converted to a tensor and normalised using mean and standard deviation of Imagenet.
class WikiArtSupervisedDataset(Dataset):
    def __init__(self, csv_path, root_dir):
        self.data=pd.read_csv(csv_path,header=None)
        self.image_path = self.data[0].tolist()
        self.labels = self.data[1].astype(int).tolist()
        self.root_dir=root_dir
        self.transform=v2.Compose([v2.RandomResizedCrop(300),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.02),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, ids):
        img_path=os.path.join(self.root_dir, self.image_path[ids])
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        img=self.transform(img)
        label =torch.tensor(self.labels[ids], dtype=torch.long)
        return img, label