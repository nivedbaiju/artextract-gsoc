import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

def load_mapping(filepath):
    mapping={}
    with open(filepath, "r") as f:
        for idx, line in enumerate(f):
            mapping[idx] = line.strip()
    return mapping


class WikiArtSupervisedDataset(Dataset):
    def __init__(self,root_dir,csv_file
    ,artist_map,genre_map,style_map,transform=None):

        self.root_dir=root_dir
        self.transform=transform

        self.data=pd.read_csv(csv_file,header=None)

        self.image_path = self.data[0].tolist()
        self.artist_labels = self.data[1].astype(int).tolist()
        self.genre_labels = self.data[2].astype(int).tolist()
        self.style_labels = self.data[3].astype(int).tolist()

        self.idx_to_artist = load_mapping(artist_map)
        self.idx_to_genre = load_mapping(genre_map)
        self.idx_to_style = load_mapping(style_map)

    def __len__(self):
        return len(self.image_path)



    def __getitem__(self, ids):
        img_path=os.path.join(self.root_dir, self.image_path[ids])

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            img = Image.new('RGB', (300,300), (0, 0, 0))

        if self.transform:
            img=self.transform(img)
        
        style=torch.tensor(self.style_labels[ids], dtype=torch.long)
        genre=torch.tensor(self.genre_labels[ids], dtype=torch.long)
        artist=torch.tensor(self.artist_labels[ids], dtype=torch.long)
        return img,style,genre,artist