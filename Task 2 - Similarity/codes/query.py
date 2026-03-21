import faiss
import numpy as np
import pickle
from extract_features import extract_feature
import matplotlib.pyplot as plt
from PIL import Image

def main():

    index=faiss.read_index("intermediates/index.faiss")

    with open("intermediates/image_paths.pkl", "rb") as f:
        paths=pickle.load(f)

    # kindly insert your image here
    query_img=r"C:\Users\nived\artextract-gsoc\Task 2 - Similarity\datasets\NAG images\0d28dd57-29c2-4e05-9c34-fb709253ef64.jpg"

    query_ftrs=extract_feature(query_img).astype("float32").reshape(1, -1)
    #we only need indexes we ignore cosine similarities
    _,I=index.search(query_ftrs, k=5)

    results=[paths[i] for i in I[0]]

    # show results
    plt.figure(figsize=(10, 3))

    plt.subplot(1, 6, 1)
    plt.imshow(Image.open(query_img))
    plt.title("Query")
    plt.axis("off")

    for i, path in enumerate(results):
        plt.subplot(1, 6, i + 2)
        plt.imshow(Image.open(path))
        plt.axis("off")

    plt.show()

if __name__=="__main__":
    main()
