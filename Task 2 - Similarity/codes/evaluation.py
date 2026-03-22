import numpy as np
import pickle
from extract_features import extract_feature
from tqdm import tqdm
import faiss
from PIL import Image

index=faiss.read_index(r"C:\Users\nived\artextract-gsoc\Task 2 - Similarity\intermediates\index.faiss")

with open(r"C:\Users\nived\artextract-gsoc\Task 2 - Similarity\intermediates\image_paths.pkl", "rb") as f:
    paths=pickle.load(f)

with open(r"C:\Users\nived\artextract-gsoc\Task 2 - Similarity\intermediates\ground_truth.pkl", "rb") as f:
    ground_truth=pickle.load(f)


def get_retrieved(query_img, k=10):
    query_ftrs=extract_feature(query_img).astype("float32").reshape(1, -1)
    D, I=index.search(query_ftrs, k=k)
    retrieved=[paths[i] for i in I[0]]
    return retrieved

#Recall@K
def recall_at_k(retrieved, gt, k):
    retrieved_k=retrieved[:k]
    hits=sum([1 for r in retrieved_k if r in gt])
    return hits / len(gt) if len(gt) > 0 else 0

#Hit@K
def hit_at_k(retrieved, gt, k):
    retrieved_k=retrieved[:k]
    return 1 if any(r in gt for r in retrieved_k) else 0

#Reciprocal Rank 
def reciprocal_rank(retrieved, gt):
    for rank, r in enumerate(retrieved, start=1):
        if r in gt:
            return 1 / rank
    return 0

recall_1=[]
recall_3=[]
recall_5=[]

hit_5=[]

mrr=[]

#call fns
for query_img, gt_list in tqdm(ground_truth.items()):

    retrieved=get_retrieved(query_img, k=10)

    recall_1.append(recall_at_k(retrieved, gt_list, 1))
    recall_3.append(recall_at_k(retrieved, gt_list, 3))
    recall_5.append(recall_at_k(retrieved, gt_list, 5))
    hit_5.append(hit_at_k(retrieved, gt_list, 5))
    mrr.append(reciprocal_rank(retrieved, gt_list))


print(f"Recall@1: {np.mean(recall_1):.4f}")
print(f"Recall@3: {np.mean(recall_3):.4f}")
print(f"Recall@5: {np.mean(recall_5):.4f}")
print(f"Hit@5: {np.mean(hit_5):.4f}")
print(f"MRR: {np.mean(mrr):.4f}")