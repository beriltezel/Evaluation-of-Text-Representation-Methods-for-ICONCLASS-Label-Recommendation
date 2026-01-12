import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer, util

DB_PATH = "Evaluation-of-Text-Representation-Methods-for-ICONCLASS-Label-Recommendation/iconclass_hierarchy.db"
LANG = "en"
TOP_K = 20
MAX_ROWS = 200000  # without the limit it'S impractical, takes too long

model = SentenceTransformer("all-MiniLM-L6-v2")

con = sqlite3.connect(DB_PATH)
cur = con.cursor()

sql = "SELECT notation, label FROM iconclass WHERE lang=?"
params = [LANG]

if MAX_ROWS is not None:
    sql += " LIMIT ?"
    params.append(MAX_ROWS)

cur.execute(sql, params)
rows = cur.fetchall()
con.close()

corpus = [r[1] for r in rows]
meta = [(r[0], r[1]) for r in rows]

corpus_embeddings = model.encode(
    corpus,
    convert_to_tensor=True,
    normalize_embeddings=True,
    batch_size=64,
    show_progress_bar=True
)

while True:
    query = input("\nQuery (empty to quit): ").strip()
    if not query:
        break

    query_embedding = model.encode(
        query,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=TOP_K)[0]

    for rank, h in enumerate(hits, start=1):
        idx = h["corpus_id"]
        score = h["score"]
        notation, label = meta[idx]
        print(f"{rank:2d}. {score:.4f}  {notation} → {label}")