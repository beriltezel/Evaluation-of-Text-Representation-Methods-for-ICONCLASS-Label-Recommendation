import os
import csv
import math
import torch
import sqlite3
import numpy as np
from collections import defaultdict
from iconclass import init
from sentence_transformers import SentenceTransformer, util

DB_PATH = "Evaluation-of-Text-Representation-Methods-for-ICONCLASS-Label-Recommendation/iconclass_hierarchy.db"
BASE_DIR = os.path.dirname(__file__)
GROUND_TRUTH_FILE = os.path.join(BASE_DIR, "ground_truth.csv")

SBERT_GROUP_DIR = "grouped_sbert_basic"
SBERT_TOP_K = 10
BM25_TOP_K = 10
LANG = "en"


#ground truth

def load_ground_truth(path):
    queries = []

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            query = row["query"].strip()
            relevant = row["relevant_notations"].strip()

            if relevant:
                relevant_set = {x.strip() for x in relevant.split("|") if x.strip()}
            else:
                relevant_set = set()

            queries.append({
                "query": query,
                "relevant": relevant_set
            })

    return queries


#metrics

def compute_metrics(retrieved, relevant, total_docs):
    retrieved = set(retrieved)
    relevant = set(relevant)

    tp = len(retrieved & relevant)
    fp = len(retrieved - relevant)
    fn = len(relevant - retrieved)
    tn = total_docs - tp - fp - fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total_docs if total_docs > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }


def evaluate_method(method_name, search_function, queries, total_docs, top_k):
    print(f"\n===== {method_name} =====")

    all_metrics = []

    for item in queries:
        query = item["query"]
        relevant = item["relevant"]

        retrieved = search_function(query, top_k=top_k)
        metrics = compute_metrics(retrieved, relevant, total_docs)

        all_metrics.append(metrics)

        print(f"\nQuery: {query}")
        print(f"Retrieved: {retrieved}")
        print(f"Relevant : {sorted(relevant)}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")
        print(f"F1       : {metrics['f1']:.4f}")
        print(f"Accuracy : {metrics['accuracy']:.4f}")

    n = len(all_metrics)

    avg_precision = sum(m["precision"] for m in all_metrics) / n
    avg_recall = sum(m["recall"] for m in all_metrics) / n
    avg_f1 = sum(m["f1"] for m in all_metrics) / n
    avg_accuracy = sum(m["accuracy"] for m in all_metrics) / n

    print(f"\n--- Average {method_name} ---")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall   : {avg_recall:.4f}")
    print(f"F1       : {avg_f1:.4f}")
    print(f"Accuracy : {avg_accuracy:.4f}")


#BM25

ic = init()

def walk(node):
    yield node
    for child in node:
        yield from walk(child)

corpus = []
for node in walk(ic):
    try:
        label = node("en").lower()
    except:
        label = ""
    tokens = label.split()
    corpus.append((node, tokens))

N = len(corpus)

df = defaultdict(int)
for _, tokens in corpus:
    for t in set(tokens):
        df[t] += 1

k1 = 1.5
b = 0.75
avgdl = sum(len(tokens) for _, tokens in corpus) / N


def bm25_score(query_tokens, doc_tokens):
    score = 0.0
    doc_len = len(doc_tokens)

    for q in query_tokens:
        f = doc_tokens.count(q)
        if f == 0:
            continue

        df_q = df.get(q, 0)
        if df_q == 0:
            continue

        idf = math.log((N - df_q + 0.5) / (df_q + 0.5) + 1)
        denom = f + k1 * (1 - b + b * (doc_len / avgdl))
        score += idf * ((f * (k1 + 1)) / denom)

    return score


def search_bm25(query, top_k=20):
    query_tokens = query.lower().split()
    results = []

    for node, tokens in corpus:
        score = bm25_score(query_tokens, tokens)
        if score > 0:
            notation = repr(node).split()[0]
            results.append((score, notation))

    results.sort(key=lambda x: x[0], reverse=True)
    return [notation for score, notation in results[:top_k]]


#SBERT

with open(os.path.join(SBERT_GROUP_DIR, "model_name.txt"), "r", encoding="utf-8") as f:
    SBERT_MODEL_NAME = f.read().strip()

sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
group_keys = np.load(os.path.join(SBERT_GROUP_DIR, "group_keys.npy"), allow_pickle=True)


def search_sbert(query, top_k=20):
    query_embedding = sbert_model.encode(
        query,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    all_hits = []

    for g in group_keys:
        g = str(g)

        emb_path = os.path.join(SBERT_GROUP_DIR, f"emb_{g}.npy")
        meta_path = os.path.join(SBERT_GROUP_DIR, f"meta_{g}.npy")

        if not os.path.exists(emb_path):
            continue

        embeddings = np.load(emb_path)
        meta = np.load(meta_path, allow_pickle=True)

        emb_tensor = torch.tensor(embeddings)

        hits = util.semantic_search(
            query_embedding,
            emb_tensor,
            top_k=top_k
        )[0]

        for h in hits:
            idx = h["corpus_id"]
            score = h["score"]
            notation, label = meta[idx]
            all_hits.append((score, notation))

    all_hits.sort(key=lambda x: x[0], reverse=True)
    return [notation for score, notation in all_hits[:top_k]]


#total docs

def get_total_docs():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM iconclass WHERE lang=?", (LANG,))
    total = cur.fetchone()[0]
    con.close()
    return total


if __name__ == "__main__":
    queries = load_ground_truth(GROUND_TRUTH_FILE)
    total_docs = get_total_docs()

    evaluate_method(
        method_name="BM25",
        search_function=search_bm25,
        queries=queries,
        total_docs=total_docs,
        top_k=BM25_TOP_K
    )

    evaluate_method(
        method_name="SBERT",
        search_function=search_sbert,
        queries=queries,
        total_docs=total_docs,
        top_k=SBERT_TOP_K
    )