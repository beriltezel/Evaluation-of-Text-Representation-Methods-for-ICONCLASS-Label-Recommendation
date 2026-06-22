import os
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer


DB_PATH = os.path.join(
    "Evaluation-of-Text-Representation-Methods-for-ICONCLASS-Label-Recommendation",
    "iconclass_hierarchy.db"
)

OUT_DIR = "sbert_data"
LANG = "en"
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64


os.makedirs(OUT_DIR, exist_ok=True)

model = SentenceTransformer(MODEL_NAME)

con = sqlite3.connect(DB_PATH)
cur = con.cursor()

cur.execute(
    """
    SELECT notation, label
    FROM iconclass
    WHERE lang=?
    ORDER BY notation
    """,
    (LANG,)
)

rows = cur.fetchall()
con.close()

meta = []
labels = []

for notation, label in rows:
    if not label:
        continue

    meta.append((notation, label))
    labels.append(label)

print(f"Number of entries to embed: {len(labels)}")

embeddings = model.encode(
    labels,
    batch_size=BATCH_SIZE,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
).astype("float32")

np.save(os.path.join(OUT_DIR, "embeddings.npy"), embeddings)
np.save(os.path.join(OUT_DIR, "meta.npy"), np.array(meta, dtype=object))

with open(os.path.join(OUT_DIR, "model_name.txt"), "w", encoding="utf-8") as f:
    f.write(MODEL_NAME + "\n")

print(f"Saved embeddings to: {os.path.join(OUT_DIR, 'embeddings.npy')}")
print(f"Saved metadata to: {os.path.join(OUT_DIR, 'meta.npy')}")
print(f"Saved model name to: {os.path.join(OUT_DIR, 'model_name.txt')}")
print("Done.")