import faiss
import numpy as np
import pickle
from extract_features import extract_feature
import matplotlib.pyplot as plt
from PIL import Image

def main():

    index=faiss.read_index(r"C:\Users\nived\artextract-gsoc\Task 2 - Similarity\intermediates\index.faiss")

    with open(r"C:\Users\nived\artextract-gsoc\Task 2 - Similarity\intermediates\image_paths.pkl", "rb") as f:
        paths=pickle.load(f)

    # kindly insert your image here
    query_img=r"C:\Users\nived\Downloads\71sx2Ah81FL.jpg"
    query_ftrs=extract_feature(query_img).astype("float32").reshape(1, -1)
    D,I=index.search(query_ftrs, k=10)

    top_score=D[0][0]

    valid_results = []
    for score, idx in zip(D[0], I[0]):
        print(f"Score: {score:.4f}, Path: {paths[idx]}")
        if score >= top_score * 0.75 and score>=0.54:
            valid_results.append((score,idx))

    valid_results=valid_results[:3]
    if len(valid_results) == 0:
        print("No valid results found.")
        return
    
    results=[paths[idx] for (_,idx) in valid_results]
    scores=[score for (score,_) in valid_results]

    # show results
    plt.figure(figsize=(3*(len(results)+1), 5))

    plt.subplot(1, len(results)+1, 1)
    plt.imshow(Image.open(query_img))
    plt.title("Query",fontsize=6)
    plt.axis("off")

    for i, (path, score) in enumerate(zip(results, scores)):
        plt.subplot(1, len(results)+1, i + 2)
        plt.imshow(Image.open(path))
        plt.title(f"Score: {score:.4f}", fontsize=8)
        plt.axis("off")

    plt.show()

if __name__=="__main__":
    main()
