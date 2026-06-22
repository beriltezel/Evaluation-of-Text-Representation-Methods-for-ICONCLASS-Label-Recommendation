import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMB_PATH = "embeddings.npy"
META_PATH = "meta.npy"
TOP_K = 20

embeddings = np.load(EMB_PATH)
meta = np.load(META_PATH, allow_pickle=True)

dim = embeddings.shape[1]

index = faiss.IndexFlatIP(dim)
index.add(embeddings)

model = SentenceTransformer("all-MiniLM-L6-v2")

while True:
    query = input("\nQuery (empty to quit): ").strip()
    if not query:
        break

    q_emb = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, ids = index.search(q_emb, TOP_K)

    for rank, idx in enumerate(ids[0], start=1):
        notation, label = meta[idx]
        print(f"{rank:2d}. {notation} → {label}")