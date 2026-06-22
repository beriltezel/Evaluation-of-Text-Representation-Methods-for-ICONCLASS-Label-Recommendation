import os
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

DB_PATH = "Evaluation-of-Text-Representation-Methods-for-ICONCLASS-Label-Recommendation/iconclass_hierarchy.db"
OUT_DIR = "grouped_sbert_basic"
LANG = "en"
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64

os.makedirs(OUT_DIR, exist_ok=True)

model = SentenceTransformer(MODEL_NAME)

con = sqlite3.connect(DB_PATH)
cur = con.cursor()

cur.execute("""
    SELECT notation, label
    FROM iconclass
    WHERE lang=?
    ORDER BY notation
""", (LANG,))

rows = cur.fetchall()
con.close()

groups = {}

for notation, label in rows:
    if not label:
        continue

    prefix = notation[:2]
    if prefix not in groups:
        groups[prefix] = []

    groups[prefix].append((notation, label))

print(f"Number of groups: {len(groups)}")

group_keys = []

for prefix, items in groups.items():
    labels = [item[1] for item in items]
    meta = [(item[0], item[1]) for item in items]

    embeddings = model.encode(
        labels,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype("float32")

    np.save(os.path.join(OUT_DIR, f"emb_{prefix}.npy"), embeddings)
    np.save(os.path.join(OUT_DIR, f"meta_{prefix}.npy"), np.array(meta, dtype=object))

    group_keys.append(prefix)

    print(f"Saved group {prefix}: {len(items)} entries")

np.save(os.path.join(OUT_DIR, "group_keys.npy"), np.array(group_keys, dtype=object))

with open(os.path.join(OUT_DIR, "model_name.txt"), "w", encoding="utf-8") as f:
    f.write(MODEL_NAME + "\n")

print("Done.")