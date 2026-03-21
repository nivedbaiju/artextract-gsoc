import os
import numpy as np
import faiss
import pickle
from tqdm import tqdm
from extract_features import extract_feature

dataset=r"C:\Users\nived\artextract-gsoc\Task 2 - Similarity\datasets\NAG images"
output_dir=r"C:\Users\nived\artextract-gsoc\Task 2 - Similarity\intermediates"
features_dir=os.path.join(output_dir, 'embeddings.npy')
images_dir=os.path.join(output_dir, 'image_paths.pkl')
index_dir=os.path.join(output_dir, 'index.faiss')

image_paths = []
embeddings=[]

for f in tqdm(sorted(os.listdir(dataset))):
    if not  f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):continue
    img_path=os.path.join(dataset,f)
    try:
        features=extract_feature(img_path)
        embeddings.append(features)
        image_paths.append(img_path)        
    except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
embeddings=np.vstack(embeddings).astype('float32')

np.save(features_dir,embeddings)

with open(images_dir,'wb') as f:
    pickle.dump(image_paths,f)

#faiss index

index=faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index,index_dir)