import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

DB_PATH = "Evaluation-of-Text-Representation-Methods-for-ICONCLASS-Label-Recommendation/iconclass_easy.db"
EMB_PATH = "embeddings.npy"
META_PATH = "meta.npy"

model = SentenceTransformer("all-MiniLM-L6-v2")

con = sqlite3.connect(DB_PATH)
cur = con.cursor()

cur.execute("SELECT notation, label FROM iconclass WHERE lang='en'")
rows = cur.fetchall()
con.close()

labels = [r[1] for r in rows]
meta = [(r[0], r[1]) for r in rows]

embeddings = model.encode(
    labels,
    batch_size=64,
    convert_to_numpy=True,
    normalize_embeddings=True
)

np.save(EMB_PATH, embeddings)
np.save(META_PATH, meta)

print(f"Saved {len(embeddings)} embeddings")