import os
import csv
import json
from datetime import datetime

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUT_DIR = "sbert_data"

GROUND_TRUTH_CSV_PATH = os.path.join(BASE_DIR,"ground_truth.csv")
OUTPUT_JSONL_PATH = os.path.join(BASE_DIR, "model_results.jsonl")

MODEL_LABEL = "sbert"
TOP_N = 20


with open(os.path.join(OUT_DIR, "model_name.txt"), "r", encoding="utf-8") as f:
    MODEL_NAME = f.read().strip()

model = SentenceTransformer(MODEL_NAME)

embeddings = np.load(os.path.join(OUT_DIR, "embeddings.npy"))
meta = np.load(os.path.join(OUT_DIR, "meta.npy"), allow_pickle=True)

emb_tensor = torch.tensor(embeddings)

valid_codes = set(str(item[0]) for item in meta)


def load_ground_truth(path=GROUND_TRUTH_CSV_PATH, has_header=True):
    
    #Loads ground truth dynamically from CSV

    #Expected format:
    #    first column  = query text
    #    other columns = relevant Iconclass codes

    #Empty cells are ignored
    #Different rows can have different numbers of relevant codes

    ground_truth = {}

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=";")

        if has_header:
            next(reader, None)

        for row in reader:
            if not row:
                continue

            query = row[0].strip()

            if not query:
                continue

            relevant_codes = [
                cell.strip()
                for cell in row[1:]
                if cell and cell.strip()
            ]

            ground_truth[query] = relevant_codes

    return ground_truth


def search_iconclass_sbert(query, top_n=TOP_N):
   
    query_embedding = model.encode(
        query,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    hits = util.semantic_search(
        query_embedding,
        emb_tensor,
        top_k=top_n
    )[0]

    results = []

    for rank, hit in enumerate(hits, start=1):
        idx = hit["corpus_id"]
        score = hit["score"]

        notation, label = meta[idx]

        results.append({
            "rank": rank,
            "code": str(notation),
            "label": str(label),
            "score": float(score)
        })

    return results


def append_jsonl(record, path=OUTPUT_JSONL_PATH):
    
    #Appends one JSON object as one line

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def check_ground_truth_against_corpus(ground_truth):
    """
    Checks whether ground-truth codes are searchable in the SBERT corpus.
    """
    missing = {}

    for query, gt_codes in ground_truth.items():
        missing_codes = [code for code in gt_codes if code not in valid_codes]

        if missing_codes:
            missing[query] = missing_codes

    return missing


def run_sbert_on_ground_truth(
    ground_truth_csv_path=GROUND_TRUTH_CSV_PATH,
    output_jsonl_path=OUTPUT_JSONL_PATH,
    top_n=TOP_N,
    has_header=True
):
    
    #Loads the ground truth, runs SBERT for every query, saves results
    
    ground_truth = load_ground_truth(
        path=ground_truth_csv_path,
        has_header=has_header
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    missing = check_ground_truth_against_corpus(ground_truth)

    if missing:
        print("\nWarning: some ground-truth codes are not in the SBERT corpus.")
        print("These codes cannot be retrieved by SBERT under the shared filtering rule.")
        print(f"Queries affected: {len(missing)}\n")

    for query, gt_codes in ground_truth.items():
        results = search_iconclass_sbert(query, top_n=top_n)

        record = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": MODEL_LABEL,
            "model_name": MODEL_NAME,
            "top_n": top_n,
            "query": query,
            "ground_truth_codes": gt_codes,
            "predicted_results": results,
            "predicted_codes": [item["code"] for item in results],
            "ground_truth_codes_missing_from_corpus": missing.get(query, [])
        }

        append_jsonl(record, path=output_jsonl_path)

    print("Finished SBERT run.")
    print(f"Queries processed: {len(ground_truth)}")
    print(f"SBERT corpus size: {len(meta)}")
    print(f"Results appended to: {output_jsonl_path}")


if __name__ == "__main__":
    run_sbert_on_ground_truth(
        ground_truth_csv_path=GROUND_TRUTH_CSV_PATH,
        output_jsonl_path=OUTPUT_JSONL_PATH,
        top_n=TOP_N,
        has_header=True
    )