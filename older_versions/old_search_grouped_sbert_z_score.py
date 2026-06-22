import os
import numpy as np
from sentence_transformers import SentenceTransformer

OUT_DIR = "grouped_sbert"
TOP_K = 20

K_STD = 1.0
MIN_GROUPS = 3
MAX_GROUPS = 10

with open(os.path.join(OUT_DIR, "model_name.txt"), "r", encoding="utf-8") as f:
    MODEL_NAME = f.read().strip()

model = SentenceTransformer(MODEL_NAME)

group_keys = np.load(os.path.join(OUT_DIR, "group_keys.npy"), allow_pickle=True)
group_centroids = np.load(os.path.join(OUT_DIR, "group_centroids.npy"))

def topk_indices(scores, k):
    if k >= len(scores):
        idx = np.argsort(-scores)
        return idx
    idx = np.argpartition(-scores, k)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx

while True:
    query = input("\nQuery (empty to quit): ").strip()
    if not query:
        break

    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")[0]

    sims = group_centroids @ q
    mu = float(sims.mean())
    sd = float(sims.std())

    if sd == 0.0:
        keep = np.argsort(-sims)[:MIN_GROUPS]
    else:
        keep = np.where(sims >= mu + K_STD * sd)[0]
        if len(keep) < MIN_GROUPS:
            keep = np.argsort(-sims)[:MIN_GROUPS]
        else:
            keep = keep[np.argsort(-sims[keep])]
            keep = keep[:MAX_GROUPS]

    all_hits = []

    for gi in keep:
        g = str(group_keys[gi])

        emb = np.load(os.path.join(OUT_DIR, f"emb_{g}.npy"))
        meta = np.load(os.path.join(OUT_DIR, f"meta_{g}.npy"), allow_pickle=True)

        scores = emb @ q
        idxs = topk_indices(scores, TOP_K)

        for idx in idxs:
            notation, label = meta[idx]
            all_hits.append((float(scores[idx]), notation, label))

    all_hits.sort(key=lambda x: x[0], reverse=True)
    all_hits = all_hits[:TOP_K]

    for rank, (score, notation, label) in enumerate(all_hits, start=1):
        print(f"{rank:2d}. {score:.4f}  {notation} → {label}")