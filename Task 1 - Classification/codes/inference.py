import torch
from model import CNN_BiLSTM
from PIL import Image
from torchvision.transforms import v2
from torch.amp import autocast
from dataset import WikiArtSupervisedDataset
from dataloader import weight_values
import os
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

#we use same preprocessing as validation for inference 
transform=v2.Compose([v2.Resize(300),v2.CenterCrop(300),
                              v2.ToTensor(),
                                v2.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])

root_dir=r"C:\Users\nived\artextract-gsoc\Task 1 - Classification\datasets\wikiart_filtered"
train_csv=r"C:\Users\nived\artextract-gsoc\Task 1 - Classification\datasets\wikiart_csv\train_labels_fixed.csv"
val_csv=r"C:\Users\nived\artextract-gsoc\Task 1 - Classification\datasets\wikiart_csv\val_labels_fixed.csv"
artist_map=r"C:\Users\nived\artextract-gsoc\Task 1 - Classification\datasets\wikiart_csv\artist_class.txt"
genre_map=r"C:\Users\nived\artextract-gsoc\Task 1 - Classification\datasets\wikiart_csv\genre_class.txt"
style_map=r"C:\Users\nived\artextract-gsoc\Task 1 - Classification\datasets\wikiart_csv\style_class.txt"
csv_path=r"C:\Users\nived\artextract-gsoc\Task 1 - Classification\datasets\wikiart_csv\train_labels_fixed.csv"

style_weights, genre_weights, artist_weights= (weight_values(csv_path=csv_path, root_dir=root_dir, artist_map=artist_map, genre_map=genre_map, style_map=style_map))
style_weights=style_weights.to(device)
genre_weights=genre_weights.to(device)
artist_weights=artist_weights.to(device)

num_style = len(style_weights)
num_genre = len(genre_weights)
num_artists = len(artist_weights)

dataset = WikiArtSupervisedDataset(
        root_dir=root_dir,
        csv_file=train_csv,
        artist_map=artist_map,
        genre_map=genre_map,
        style_map=style_map,
        transform=None
    )

idx_to_style = dataset.idx_to_style
idx_to_genre = dataset.idx_to_genre
idx_to_artist = dataset.idx_to_artist

model=CNN_BiLSTM(num_style=num_style,num_genre=num_genre,num_artists=num_artists).to(device)
checkpoint = torch.load("..\\checkpoints/best_model.pth", map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()

def predict(image_path):
    if not os.path.exists(image_path):
        print("Image not found:", image_path)
        return

    with Image.open(image_path) as img:
        img=img.convert("RGB")

    img=transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        with autocast(device_type=device.type):
            style_out, genre_out, artist_out = model(img)
        style_pred= torch.argmax(style_out, dim=1).item()
        genre_pred= torch.argmax(genre_out, dim=1).item()
        artist_pred= torch.argmax(artist_out, dim=1).item()

        style_prob = torch.softmax(style_out, dim=1)[0, style_pred].item()
        genre_prob = torch.softmax(genre_out, dim=1)[0, genre_pred].item()
        artist_prob = torch.softmax(artist_out, dim=1)[0, artist_pred].item()

    style_label = idx_to_style[style_pred]
    genre_label = idx_to_genre[genre_pred]
    artist_label = idx_to_artist[artist_pred] 

    print(f"Predicted Style: {style_label} (Index={style_pred})(Confidence={style_prob:.4f})")
    print(f"Predicted Genre: {genre_label} (Index={genre_pred})(Confidence={genre_prob:.4f})")
    print(f"Predicted Artist: {artist_label} (Index={artist_pred})(Confidence={artist_prob:.4f})")

predict(r"C:\Users\nived\Downloads\pablo-picasso-three-musicians.jpg")