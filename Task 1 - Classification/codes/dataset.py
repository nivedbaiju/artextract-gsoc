import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class WikiArtSupervisedDataset(Dataset):
    def __init__(self, csv_path, root_dir):
        self.data=pd.read_csv(csv_path,header=None)
        self.image_path = self.data[0].tolist()
        self.labels = self.data[1].astype(int).tolist()
        self.root_dir=root_dir


    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, ids):
        img_path=os.path.join(self.root_dir, self.image_path[ids])
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        img=self.transform(img)
        label =torch.tensor(self.labels[ids], dtype=torch.long)
        return img, label