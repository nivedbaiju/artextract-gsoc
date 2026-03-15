import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from dataset import WikiArtSupervisedDataset
from dataloader import weight_values
from model import CNN_BiLSTM
from torchvision.transforms import v2

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# paths
root_dir=r"C:\Users\nived\artextract-gsoc\Task 1 - Classification\datasets\wikiart_filtered"
train_csv=r"C:\Users\nived\artextract-gsoc\Task 1 - Classification\datasets\wikiart_csv\train_labels_fixed.csv"
val_csv=r"C:\Users\nived\artextract-gsoc\Task 1 - Classification\datasets\wikiart_csv\val_labels_fixed.csv"
artist_map=r"C:\Users\nived\artextract-gsoc\Task 1 - Classification\datasets\wikiart_csv\artist_class.txt"
genre_map=r"C:\Users\nived\artextract-gsoc\Task 1 - Classification\datasets\wikiart_csv\genre_class.txt"
style_map=r"C:\Users\nived\artextract-gsoc\Task 1 - Classification\datasets\wikiart_csv\style_class.txt"
csv_path=r"C:\Users\nived\artextract-gsoc\Task 1 - Classification\datasets\wikiart_csv\train_labels_fixed.csv"

checkpoint_path=r"C:\Users\nived\artextract-gsoc\Task 1 - Classification\checkpoints\best_model.pth"


#inorder to detect outliers we will use knn + confidence score.
k= 20
outlier_percentage= 0.02

val_transform=v2.Compose([v2.Resize(300),v2.CenterCrop(300),
                              v2.ToTensor(),
                                v2.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# dataset
dataset=WikiArtSupervisedDataset(root_dir, csv_path,artist_map,genre_map,style_map, transform=val_transform)
loader=DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)

style_weights, genre_weights, artist_weights= (weight_values(csv_path=csv_path, root_dir=root_dir, artist_map=artist_map, genre_map=genre_map, style_map=style_map))
style_weights=style_weights.to(device)
genre_weights=genre_weights.to(device)
artist_weights=artist_weights.to(device)

num_style = len(style_weights)
num_genre = len(genre_weights)
num_artists = len(artist_weights)

# define model
model = CNN_BiLSTM(num_style, num_genre, num_artists)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model"])
model = model.to(device)
model.eval()


embeddings=[]
style_labels= []
genre_labels= []
artist_labels= []
style_confidence= []
genre_confidence= []
artist_confidence= []
paths= []


with torch.no_grad():
    for batch_idx, (images, style, genre, artist) in enumerate(tqdm(loader)):

      images = images.to(device)

      style_logits, genre_logits, artist_logits, features = model(images, return_features=True)

      embeddings.append(features.cpu().numpy())

      style_labels.extend(style.numpy())
      genre_labels.extend(genre.numpy())
      artist_labels.extend(artist.numpy())

      start = batch_idx * loader.batch_size
      end = start + images.size(0)

      batch_paths = dataset.data.iloc[start:end, 0].values
      paths.extend(batch_paths)

      style_prob = F.softmax(style_logits, dim=1)
      genre_prob = F.softmax(genre_logits, dim=1)
      artist_prob = F.softmax(artist_logits, dim=1)

      style_confidence.extend(style_prob.max(dim=1).values.cpu().numpy())
      genre_confidence.extend(genre_prob.max(dim=1).values.cpu().numpy())
      artist_confidence.extend(artist_prob.max(dim=1).values.cpu().numpy())


embeddings = np.concatenate(embeddings)
embeddings=embeddings/np.linalg.norm(embeddings,axis=1, keepdims=True)

neighbours= NearestNeighbors(n_neighbors=k+1,metric="cosine")
neighbours.fit(embeddings)

distances, indices= neighbours.kneighbors(embeddings)

def neighbor_agreement(labels):
    agreements = []
    for i in range(len(labels)):
        neighbor_idx = indices[i][1:]  # exclude itself
        neighbor_labels = labels[neighbor_idx]

        agree = np.mean(neighbor_labels == labels[i])
        agreements.append(agree)

    return np.array(agreements)


style_agree = neighbor_agreement(np.array(style_labels))
genre_agree = neighbor_agreement(np.array(genre_labels))
artist_agree = neighbor_agreement(np.array(artist_labels))


# convert to confidence arrays
style_conf =np.array(style_confidence)
genre_conf = np.array(genre_confidence)
artist_conf= np.array(artist_confidence)


# Outlier score
score = ((1-style_agree)+(1-style_conf) + (1-genre_agree)+(1-genre_conf) + (1-artist_agree)+(1-artist_conf))


# Select top outlier% outliers
n_outliers = int(len(score) * outlier_percentage)
outlier_idx = np.argsort(score)[-n_outliers:]

results = []

for i in outlier_idx:
    results.append({
        "path": paths[i],
        "style_label": style_labels[i],
        "genre_label": genre_labels[i],
        "artist_label": artist_labels[i],
        "score": score[i],
        "style_agree": style_agree[i],
        "genre_agree": genre_agree[i],
        "artist_agree": artist_agree[i],
        "style_conf": style_conf[i],
        "genre_conf": genre_conf[i],
        "artist_conf": artist_conf[i],
    })


df = pd.DataFrame(results)
df = df.sort_values("score", ascending=False)
#save detected outliers to csv file..
df.to_csv(r"..\\results/detected_outliers.csv", index=False)
