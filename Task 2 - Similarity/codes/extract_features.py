import torch
import clip
import timm
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as T


if torch.cuda.is_available():device='cuda'
else: device='cpu'
print(device)
# i will be using both clip and dino for extracting image features.
# clip will semantically understand the image and dino will capture the visual features of the image and both the features are finally combined.

clip_model,clip_preprocessing=clip.load('ViT-L/14',device=device)

dino_model=timm.create_model('vit_large_patch14_dinov2', pretrained=True,num_classes=0)

dino_model.eval()
dino_model.to(device)

dino_transform = T.Compose([T.Resize(518),T.CenterCrop(518),
    T.ToTensor(),T.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),])

def load_image(image_path):
    return Image.open(image_path).convert('RGB')

def clip_features(img):
    with torch.no_grad():
        img=clip_preprocessing(img).unsqueeze(0).to(device)
        features=clip_model.encode_image(img)
        features=F.normalize(features,dim=-1)
    return features.squeeze().cpu().numpy()

def dino_features(img):
    with torch.no_grad():
        img=dino_transform(img).unsqueeze(0).to(device)
        features=dino_model(img)
        features=F.normalize(features,dim=-1)
    return features.squeeze().cpu().numpy()

def combine_features(clip_ftrs, dino_ftrs,alpha=0.5):
    clip_ftrs/=np.linalg.norm(clip_ftrs)
    dino_ftrs/=np.linalg.norm(dino_ftrs)

    combined_ftrs=np.concatenate([alpha*clip_ftrs,(1-alpha)*dino_ftrs])
    combined_ftrs/=np.linalg.norm(combined_ftrs)
    return combined_ftrs.astype('float32')

#main fn
def extract_feature(image_path,alpha=0.5):
    img=load_image(image_path)
    clip_ftrs=clip_features(img)
    dino_ftrs=dino_features(img)
    combined_ftrs=combine_features(clip_ftrs,dino_ftrs,alpha)
    return combined_ftrs


