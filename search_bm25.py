from iconclass import init
import math
from collections import defaultdict

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


def search_iconclass_bm25(query):
    query_tokens = query.lower().split()
    results = []

    for node, tokens in corpus:
        score = bm25_score(query_tokens, tokens)
        if score > 0:
            results.append((score, node))

    results.sort(key=lambda x: x[0], reverse=True)
    return results


if __name__ == "__main__":
    query = input("Enter your Iconclass search query: ").strip()

    results = search_iconclass_bm25(query)

    print("\nTop results:\n")

    for score, node in results[:20]:
        notation = repr(node).split()[0]
        try:
            label = node("en")
        except:
            label = "(no label)"
        print(f"{score:.4f}  {notation} → {label}")