import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

OUT_DIR = "grouped_sbert_basic"
TOP_K = 20

with open(os.path.join(OUT_DIR, "model_name.txt"), "r", encoding="utf-8") as f:
    MODEL_NAME = f.read().strip()

model = SentenceTransformer(MODEL_NAME)
group_keys = np.load(os.path.join(OUT_DIR, "group_keys.npy"), allow_pickle=True)

while True:
    query = input("\nQuery (empty to quit): ").strip()
    if not query:
        break

    query_embedding = model.encode(
        query,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    all_hits = []

    for g in group_keys:
        g = str(g)

        emb_path = os.path.join(OUT_DIR, f"emb_{g}.npy")
        meta_path = os.path.join(OUT_DIR, f"meta_{g}.npy")

        if not os.path.exists(emb_path):
            continue

        embeddings = np.load(emb_path)
        meta = np.load(meta_path, allow_pickle=True)

        emb_tensor = torch.tensor(embeddings)

        hits = util.semantic_search(
            query_embedding,
            emb_tensor,
            top_k=TOP_K
        )[0]

        for h in hits:
            idx = h["corpus_id"]
            score = h["score"]
            notation, label = meta[idx]
            all_hits.append((score, notation, label))

    all_hits.sort(key=lambda x: x[0], reverse=True)
    all_hits = all_hits[:TOP_K]

    for rank, (score, notation, label) in enumerate(all_hits, start=1):
        print(f"{rank:2d}. {score:.4f}  {notation} → {label}")