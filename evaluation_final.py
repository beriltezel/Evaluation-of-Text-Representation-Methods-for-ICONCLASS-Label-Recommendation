import os
import json
import sqlite3
from datetime import datetime
from collections import defaultdict

import matplotlib.pyplot as plt

#This evaluation needs these files to be present in the same folder in order to function properly:
# -> iconclass_hierarchy.db
# -> ground_truth.csv
# -> model_results.jsonl


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_RESULTS_JSONL_PATH = os.path.join(BASE_DIR, "model_results.jsonl")
HIERARCHY_DB_PATH = os.path.join(BASE_DIR, "iconclass_hierarchy.db")
EVALUATION_RESULTS_JSONL_PATH = os.path.join(BASE_DIR, "evaluation_results.jsonl")

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
GRAPH_DIR = os.path.join(BASE_DIR, "evaluation_graphs", RUN_TIMESTAMP)


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def precision_recall_f1(predicted_codes, ground_truth_codes):
    predicted_set = set(predicted_codes)
    ground_truth_set = set(ground_truth_codes)

    if not predicted_set:
        precision = 0.0
    else:
        precision = len(predicted_set & ground_truth_set) / len(predicted_set)

    if not ground_truth_set:
        recall = 0.0
    else:
        recall = len(predicted_set & ground_truth_set) / len(ground_truth_set)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def r_precision(predicted_codes, ground_truth_codes):
    r = len(ground_truth_codes)

    if r == 0:
        return 0.0

    top_r = predicted_codes[:r]
    return len(set(top_r) & set(ground_truth_codes)) / r


def average_precision(predicted_codes, ground_truth_codes):
    ground_truth_set = set(ground_truth_codes)

    if not ground_truth_set:
        return 0.0

    hits = 0
    precision_sum = 0.0
    already_found = set()

    for rank, code in enumerate(predicted_codes, start=1):
        if code in ground_truth_set and code not in already_found:
            hits += 1
            already_found.add(code)
            precision_sum += hits / rank

    return precision_sum / len(ground_truth_set)


class IconclassHierarchy:
    def __init__(self, db_path):
        self.con = sqlite3.connect(db_path)
        self.cache = {}

    def get_parent_depth(self, code):
        if code in self.cache:
            return self.cache[code]

        cur = self.con.cursor()
        cur.execute(
            """
            SELECT parent, depth
            FROM iconclass
            WHERE notation=?
            LIMIT 1
            """,
            (code,)
        )

        row = cur.fetchone()

        if row is None:
            self.cache[code] = None
            return None

        parent, depth = row
        self.cache[code] = (parent, depth)
        return parent, depth

    def ancestors(self, code):
        result = {}
        current = code

        while current:
            parent_depth = self.get_parent_depth(current)

            if parent_depth is None:
                break

            parent, depth = parent_depth
            result[current] = depth
            current = parent

        return result

    def wu_palmer(self, code1, code2):
        if code1 == code2:
            return 1.0

        info1 = self.get_parent_depth(code1)
        info2 = self.get_parent_depth(code2)

        if info1 is None or info2 is None:
            return 0.0

        _, depth1 = info1
        _, depth2 = info2

        ancestors1 = self.ancestors(code1)
        ancestors2 = self.ancestors(code2)

        common = set(ancestors1.keys()) & set(ancestors2.keys())

        if not common:
            return 0.0

        lca = max(common, key=lambda c: ancestors1[c])
        lca_depth = ancestors1[lca]

        depth1 = depth1 + 1
        depth2 = depth2 + 1
        lca_depth = lca_depth + 1

        return (2 * lca_depth) / (depth1 + depth2)

    def close(self):
        self.con.close()


def average_best_wu_palmer(predicted_codes, ground_truth_codes, hierarchy):
    if not predicted_codes or not ground_truth_codes:
        return {
            "wu_palmer_gt_to_pred": 0.0,
            "wu_palmer_pred_to_gt": 0.0,
            "wu_palmer_mean": 0.0
        }

    gt_to_pred_scores = []

    for gt_code in ground_truth_codes:
        best_score = max(
            hierarchy.wu_palmer(gt_code, pred_code)
            for pred_code in predicted_codes
        )
        gt_to_pred_scores.append(best_score)

    pred_to_gt_scores = []

    for pred_code in predicted_codes:
        best_score = max(
            hierarchy.wu_palmer(pred_code, gt_code)
            for gt_code in ground_truth_codes
        )
        pred_to_gt_scores.append(best_score)

    gt_to_pred_avg = sum(gt_to_pred_scores) / len(gt_to_pred_scores)
    pred_to_gt_avg = sum(pred_to_gt_scores) / len(pred_to_gt_scores)

    return {
        "wu_palmer_gt_to_pred": gt_to_pred_avg,
        "wu_palmer_pred_to_gt": pred_to_gt_avg,
        "wu_palmer_mean": (gt_to_pred_avg + pred_to_gt_avg) / 2
    }


def evaluate_record(record, hierarchy):
    predicted_codes = record.get("predicted_codes", [])
    ground_truth_codes = record.get("ground_truth_codes", [])

    precision, recall, f1 = precision_recall_f1(
        predicted_codes,
        ground_truth_codes
    )

    wu_palmer_scores = average_best_wu_palmer(
        predicted_codes,
        ground_truth_codes,
        hierarchy
    )

    evaluation = {
        "evaluation_timestamp": datetime.now().isoformat(timespec="seconds"),
        "source_run_id": record.get("run_id"),
        "model": record.get("model"),
        "model_name": record.get("model_name"),
        "query": record.get("query"),
        "top_n": record.get("top_n"),
        "ground_truth_codes": ground_truth_codes,
        "predicted_codes": predicted_codes,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "r_precision": r_precision(predicted_codes, ground_truth_codes),
        "map": average_precision(predicted_codes, ground_truth_codes),
        "wu_palmer_gt_to_pred": wu_palmer_scores["wu_palmer_gt_to_pred"],
        "wu_palmer_pred_to_gt": wu_palmer_scores["wu_palmer_pred_to_gt"],
        "wu_palmer_mean": wu_palmer_scores["wu_palmer_mean"],
        "ground_truth_codes_missing_from_corpus": record.get(
            "ground_truth_codes_missing_from_corpus",
            []
        )
    }

    return evaluation


def summarize_by_model(records):
    metrics = [
        "precision",
        "recall",
        "f1",
        "r_precision",
        "map",
        "wu_palmer_mean",
        "wu_palmer_gt_to_pred",
        "wu_palmer_pred_to_gt"
    ]

    scores_by_model = defaultdict(lambda: defaultdict(list))

    for record in records:
        model = record.get("model")

        if not model:
            continue

        for metric in metrics:
            value = record.get(metric)

            if value is not None:
                scores_by_model[model][metric].append(value)

    models = sorted(scores_by_model.keys())
    summary = {}

    for model in models:
        summary[model] = {}

        for metric in metrics:
            values = scores_by_model[model][metric]

            if values:
                summary[model][metric] = sum(values) / len(values)
            else:
                summary[model][metric] = 0.0

    return summary, metrics, models


def plot_evaluation_results(records):
    os.makedirs(GRAPH_DIR, exist_ok=True)

    summary, metrics, models = summarize_by_model(records)

    if not models:
        print("No model scores found for plotting.")
        return

    for metric in metrics:
        values = [summary[model][metric] for model in models]

        plt.figure(figsize=(8, 5))
        plt.bar(models, values)
        plt.ylim(0, 1)
        plt.title(f"Average {metric} by model")
        plt.xlabel("Model")
        plt.ylabel(metric)
        plt.tight_layout()

        graph_path = os.path.join(GRAPH_DIR, f"{metric}_by_model.png")
        plt.savefig(graph_path, dpi=300)

    x = range(len(models))
    width = 0.1

    plt.figure(figsize=(12, 6))

    for i, metric in enumerate(metrics):
        values = [summary[model][metric] for model in models]
        positions = [pos + (i - len(metrics) / 2) * width for pos in x]
        plt.bar(positions, values, width=width, label=metric)

    plt.xticks(list(x), models)
    plt.ylim(0, 1)
    plt.title("Average evaluation metrics by model")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()

    combined_graph_path = os.path.join(GRAPH_DIR, "all_metrics_by_model.png")
    plt.savefig(combined_graph_path, dpi=300)

    print("Graphs saved in:")
    print(GRAPH_DIR)

    plt.show()

def plot_query_performance(records):
    os.makedirs(GRAPH_DIR, exist_ok=True)

# This function creates graphs showing metric performance for each query, with one graph per metric.

    metrics = [
        "precision",
        "recall",
        "f1",
        "r_precision",
        "map",
        "wu_palmer_mean",
        "wu_palmer_gt_to_pred",
        "wu_palmer_pred_to_gt"
    ]

    query_order = []
    model_order = []
    values_by_model_query = {}

    for record in records:
        model = record.get("model")
        query = record.get("query")

        if not model or not query:
            continue

        if query not in query_order:
            query_order.append(query)

        if model not in model_order:
            model_order.append(model)

        values_by_model_query[(model, query)] = record

    if not query_order or not model_order:
        print("No query scores found for plotting.")
        return

    x = range(len(query_order))

    for metric in metrics:
        plt.figure(figsize=(16, 7))

        for model in model_order:
            values = []

            for query in query_order:
                record = values_by_model_query.get((model, query))

                if record is None:
                    values.append(None)
                else:
                    values.append(record.get(metric, 0.0))

            plt.plot(
                x,
                values,
                marker="o",
                markersize=4,
                linewidth=1.5,
                label=model
            )

        plt.xticks(list(x), query_order, rotation=75, ha="right")
        plt.ylim(0, 1.05)
        plt.title(f"{metric} per query")
        plt.xlabel("Query")
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        graph_path = os.path.join(GRAPH_DIR, f"{metric}_per_query.png")
        plt.savefig(graph_path, dpi=300)

    print("Query performance graphs saved in:")
    print(GRAPH_DIR)

    plt.show()

def run_evaluation():
    model_records = load_jsonl(MODEL_RESULTS_JSONL_PATH)
    hierarchy = IconclassHierarchy(HIERARCHY_DB_PATH)

    evaluation_records = []

    for record in model_records:
        evaluation = evaluate_record(record, hierarchy)
        evaluation_records.append(evaluation)

    hierarchy.close()

    write_jsonl(evaluation_records, EVALUATION_RESULTS_JSONL_PATH)

    print("Finished evaluation.")
    print(f"Model result records processed: {len(model_records)}")
    print(f"Evaluation results written to: {EVALUATION_RESULTS_JSONL_PATH}")

    plot_evaluation_results(evaluation_records)
    plot_query_performance(evaluation_records)

if __name__ == "__main__":
    run_evaluation()