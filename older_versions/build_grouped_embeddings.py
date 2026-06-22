import os
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

DB_PATH = "Evaluation-of-Text-Representation-Methods-for-ICONCLASS-Label-Recommendation/iconclass_hierarchy.db"
OUT_DIR = "grouped_sbert"
LANG = "en"

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64

os.makedirs(OUT_DIR, exist_ok=True)

model = SentenceTransformer(MODEL_NAME)

def save_group(prefix, group_meta, group_labels, group_centroids, group_keys):
    if not group_labels:
        return

    emb = model.encode(
        group_labels,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype("float32")

    meta_arr = np.array(group_meta, dtype=object)

    np.save(os.path.join(OUT_DIR, f"emb_{prefix}.npy"), emb)
    np.save(os.path.join(OUT_DIR, f"meta_{prefix}.npy"), meta_arr)

    centroid = emb.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

    group_keys.append(prefix)
    group_centroids.append(centroid)

    print(prefix, len(group_labels))


con = sqlite3.connect(DB_PATH)
cur = con.cursor()

cur.execute("""
    SELECT notation, label
    FROM iconclass
    WHERE lang=?
    ORDER BY notation
""", (LANG,))

group_keys = []
group_centroids = []

current_prefix = None
group_labels = []
group_meta = []

n = 0

for notation, label in cur:
    if not label:
        continue

    prefix = (notation or "")[:2]
    if len(prefix) < 2:
        continue

    if current_prefix is None:
        current_prefix = prefix

    if prefix != current_prefix:
        save_group(current_prefix, group_meta, group_labels, group_centroids, group_keys)
        current_prefix = prefix
        group_labels = []
        group_meta = []

    group_labels.append(label)
    group_meta.append((notation, label))

    n += 1
    if n % 200000 == 0:
        print("rows:", n)

con.close()

save_group(current_prefix, group_meta, group_labels, group_centroids, group_keys)

np.save(os.path.join(OUT_DIR, "group_keys.npy"), np.array(group_keys, dtype=object))
np.save(os.path.join(OUT_DIR, "group_centroids.npy"), np.vstack(group_centroids).astype("float32"))

with open(os.path.join(OUT_DIR, "model_name.txt"), "w", encoding="utf-8") as f:
    f.write(MODEL_NAME + "\n")

print("done")