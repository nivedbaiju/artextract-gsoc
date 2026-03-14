from dataloader import getdataloader,weight_values
from model import CNN_BiLSTM
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast,GradScaler
import matplotlib.pyplot as plt
import os

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)

print(f"Using device:{device}")

epochs=25

root_dir=r"C:\Users\nived\artextract-gsoc\Task 1 - Classification\datasets\wikiart_filtered"
train_csv=r"C:\Users\nived\Downloads\wikiart_csv\train_labels_merged.csv"
val_csv=r"C:\Users\nived\Downloads\wikiart_csv\val_labels_merged.csv"
artist_map=r"C:\Users\nived\Downloads\wikiart_csv\artist_class.txt"
genre_map=r"C:\Users\nived\Downloads\wikiart_csv\genre_class.txt"
style_map=r"C:\Users\nived\Downloads\wikiart_csv\style_class.txt"
csv_path=r"C:\Users\nived\Downloads\wikiart_csv\train_labels_merged.csv"

train_loader,val_loader=getdataloader(train_csv=train_csv,val_csv=val_csv,root_dir=root_dir,artist_map=artist_map,genre_map=genre_map,style_map=style_map)


style_weights, genre_weights, artist_weights= (weight_values(csv_path=csv_path, root_dir=root_dir, artist_map=artist_map, genre_map=genre_map, style_map=style_map))
style_weights=style_weights.to(device)
genre_weights=genre_weights.to(device)
artist_weights=artist_weights.to(device)

num_style = len(style_weights)
num_genre = len(genre_weights)
num_artists = len(artist_weights)

model=CNN_BiLSTM(num_style=num_style,num_genre=num_genre,num_artists=num_artists).to(device)

for p in model.cnn_backbone.parameters():
    p.requires_grad = False

style_loss_fn=nn.CrossEntropyLoss(weight=style_weights,label_smoothing=0.1)
genre_loss_fn=nn.CrossEntropyLoss(weight=genre_weights,label_smoothing=0.1)
artist_loss_fn=nn.CrossEntropyLoss(weight=artist_weights,label_smoothing=0.1)
optimizer=optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)

#we use cosine decay lr scheduler.
scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)

#use GradScaler to prevent underflow
scaler=GradScaler(enabled=(device.type == "cuda"))

best_val_acc=0.0
w_style,w_genre,w_artist=num_style**-0.5,num_genre**-0.5,num_artists**-0.5
total_w= w_style+ w_genre+ w_artist
w_style/=total_w
w_genre/=total_w   
w_artist/=total_w

#we log these metrics for visualizaton
train_losses = []
val_losses = []
style_accs = []
genre_accs = []
artist_accs = []

#training loop
for epoch in range(epochs):
    if epoch == 3:
        for p in model.cnn_backbone.parameters():
            p.requires_grad = True
    model.train()
    total_loss=0.0

    for images,style_labels,genre_labels,artist_labels in train_loader:
        images=images.to(device)
        style_labels=style_labels.to(device)
        genre_labels=genre_labels.to(device)
        artist_labels=artist_labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type):
            style_out,genre_out,artist_out=model(images)
            style_loss= style_loss_fn(style_out, style_labels)
            genre_loss= genre_loss_fn(genre_out, genre_labels)
            artist_loss= artist_loss_fn(artist_out, artist_labels)

            #final loss will be weighted sum of all three losses
            loss= w_style*style_loss+ w_genre*genre_loss+ w_artist*artist_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
    scheduler.step()

    avg_loss= total_loss / len(train_loader)
    train_losses.append(avg_loss)
    lr=optimizer.param_groups[0]['lr']
    print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_loss:.4f}, LR:{lr:.6f}")

    #validation loop

    model.eval()
    validation_loss=0.0
    correct_style=0
    correct_genre=0
    correct_artist=0
    total_samples=0

    #follows almost same logic as training loop but we dont backprop on validation data.
    with torch.no_grad():
        for images,style_labels,genre_labels,artist_labels in val_loader:
            images=images.to(device)
            style_labels=style_labels.to(device)
            genre_labels=genre_labels.to(device)
            artist_labels=artist_labels.to(device)

            with autocast(device_type=device.type):
                style_out,genre_out,artist_out=model(images)

                style_loss= style_loss_fn(style_out, style_labels)
                genre_loss= genre_loss_fn(genre_out, genre_labels)
                artist_loss= artist_loss_fn(artist_out, artist_labels)

                loss= w_style*style_loss+w_genre*genre_loss+ w_artist*artist_loss
            validation_loss+=loss.item()

            _, predicted_style = torch.max(style_out, 1)
            _, predicted_genre = torch.max(genre_out, 1)
            _, predicted_artist = torch.max(artist_out, 1)

            correct_style += (predicted_style == style_labels).sum().item()
            correct_genre += (predicted_genre == genre_labels).sum().item()
            correct_artist += (predicted_artist == artist_labels).sum().item()
            total_samples += style_labels.size(0)

        avg_val_loss=validation_loss/len(val_loader)
        style_acc= correct_style/total_samples
        genre_acc= correct_genre/total_samples
        artist_acc= correct_artist/total_samples

        val_losses.append(avg_val_loss)
        style_accs.append(style_acc)
        genre_accs.append(genre_acc)
        artist_accs.append(artist_acc)

        combined_accuracy=(w_style*style_acc+ w_genre*genre_acc+ w_artist*artist_acc)
        #we save the best model
        if combined_accuracy > best_val_acc:
            best_val_acc= combined_accuracy
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print(f"New best model obtained with combined accuracy: {best_val_acc:.4f}")
        
        print(f"Epoch [{epoch+1}/{epochs}] | "
            f"Validation Loss: {avg_val_loss:.4f} | "
            f"Style Acc: {style_acc:.4f} | "
            f"Genre Acc: {genre_acc:.4f} | "
            f"Artist Acc: {artist_acc:.4f}")
        

#plot the metrics

#loss curve
epochs_range= range(1, epochs+1)
plt.figure()
plt.plot(epochs_range,train_losses, label="Training Loss")
plt.plot(epochs_range, val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.savefig("results/loss_curve.png")
plt.close()

#accuracy curves
plt.figure()
plt.plot(epochs_range,style_accs, label="Style Accuracy")
plt.plot(epochs_range,genre_accs, label="Genre Accuracy")
plt.plot(epochs_range,artist_accs, label="Artist Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy per Task")
plt.legend()
plt.savefig("results/accuracy_curve.png")
plt.close()
