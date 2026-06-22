import os
import sqlite3
from functools import lru_cache
import time

DB_PATH = os.path.join("Evaluation-of-Text-Representation-Methods-for-ICONCLASS-Label-Recommendation", "iconclass_hierarchy.db")

def prefix_bonus(base_notation: str, candidate_notation: str) -> int:
    # example: 31C21 vs 31C214 → bonus
    match_len = 0
    for a, b in zip(base_notation, candidate_notation):
        if a == b:
            match_len += 1
        else:
            break
    return match_len * 2


def text_match_score(label: str, notation: str, query_words) -> int:
    score = 0
    words = label.lower().split()
    label_l = label.lower()
    notation_l = notation.lower()

    for q in query_words:
        ql = q.lower()

        # exact word
        if ql in words:
            score += 50
            continue

        # substring match
        if ql in label_l:
            score += 20

        # match in notation
        if ql in notation_l:
            score += 10

    return score


def search_iconclass_classic_sqlite(
    query: str,
    lang: str = "en",
    limit: int = 20,
    strong_limit: int = 200,         # reduced default
    neighborhood_steps: int = 3,     # reduced default
    seed_k: int = 50                # cap comparisons to top-K seeds
):
    """
    SQLite/FTS5 rewrite of easy search.

    Pipeline:
    1) strong matches via FTS + notation LIKE
    2) expand candidate set around strong matches (ancestors/descendants)
    3) score candidates with:
        - text_match_score
        - tree distance bonus
        - prefix bonus
    """

    query = query.strip()
    if not query:
        return []

    query_words = query.lower().split()

    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row

        # Strong matches
        strong = con.execute("""
            SELECT notation, label
            FROM iconclass_fts
            WHERE iconclass_fts MATCH ?
              AND lang = ?
            LIMIT ?;
        """, (query, lang, strong_limit)).fetchall()

        # notation substring matches
        like = f"%{query.lower()}%"
        code_hits = con.execute("""
            SELECT notation, label
            FROM iconclass
            WHERE lang = ?
              AND lower(notation) LIKE ?
            LIMIT ?;
        """, (lang, like, strong_limit)).fetchall()

        strong_map = {}
        for r in list(strong) + list(code_hits):
            strong_map[r["notation"]] = r["label"]

        if not strong_map:
            return []

        strong_notations = list(strong_map.keys())
        seed_notations = strong_notations[: min(len(strong_notations), seed_k)]

        # Expanding candidates around seeds (avoiding scanning whole DB)
        candidates = dict(strong_map)  # notation -> label

        # expanding around a limited number of seeds
        expand_seeds = seed_notations[: min(len(seed_notations), 50)]

        for seed in expand_seeds:
            neigh = con.execute("""
                WITH RECURSIVE
                up(n, steps) AS (
                    SELECT notation, 0
                    FROM iconclass
                    WHERE lang=? AND notation=?
                    UNION ALL
                    SELECT i.parent, steps+1
                    FROM iconclass i
                    JOIN up ON up.n=i.notation
                    WHERE i.lang=? AND i.parent IS NOT NULL AND steps < ?
                ),
                down(n, steps) AS (
                    SELECT notation, 0
                    FROM iconclass
                    WHERE lang=? AND notation=?
                    UNION ALL
                    SELECT c.notation, steps+1
                    FROM iconclass c
                    JOIN down ON c.parent=down.n
                    WHERE c.lang=? AND steps < ?
                )
                SELECT DISTINCT i.notation, i.label
                FROM iconclass i
                WHERE i.lang=?
                  AND i.notation IN (SELECT n FROM up UNION SELECT n FROM down)
                LIMIT 5000;
            """, (lang, seed, lang, neighborhood_steps,
                  lang, seed, lang, neighborhood_steps,
                  lang)).fetchall()

            for r in neigh:
                candidates.setdefault(r["notation"], r["label"])

        cand_notations = list(candidates.keys())

        # Loading depths for candidates
        placeholders = ",".join("?" * len(cand_notations))
        rows = con.execute(f"""
            SELECT notation, depth
            FROM iconclass
            WHERE lang=? AND notation IN ({placeholders});
        """, [lang, *cand_notations]).fetchall()

        depth = {r["notation"]: r["depth"] for r in rows}

        # Memory-safe LCA + distance
        @lru_cache(maxsize=200_000)
        def lca_depth(a: str, b: str) -> int:
            a_anc = {}

            cur = con.execute("""
                WITH RECURSIVE anc(n, d) AS (
                    SELECT notation, depth
                    FROM iconclass
                    WHERE lang=? AND notation=?
                    UNION ALL
                    SELECT i.parent, p.depth
                    FROM iconclass i
                    JOIN anc ON anc.n=i.notation
                    JOIN iconclass p ON p.lang=i.lang AND p.notation=i.parent
                    WHERE i.lang=? AND i.parent IS NOT NULL
                )
                SELECT n, d FROM anc;
            """, (lang, a, lang))

            for n, d in cur:
                a_anc[n] = d

            cur = con.execute("""
                WITH RECURSIVE anc(n) AS (
                    SELECT notation
                    FROM iconclass
                    WHERE lang=? AND notation=?
                    UNION ALL
                    SELECT i.parent
                    FROM iconclass i
                    JOIN anc ON anc.n=i.notation
                    WHERE i.lang=? AND i.parent IS NOT NULL
                )
                SELECT n FROM anc;
            """, (lang, b, lang))

            for (n,) in cur:
                if n in a_anc:
                    return a_anc[n]

            return 0

        @lru_cache(maxsize=200_000)
        def tree_distance(a: str, b: str) -> int:
            da = depth.get(a, 0)
            db = depth.get(b, 0)
            dlca = lca_depth(a, b)
            return da + db - 2 * dlca

        # Scoring candidates
        scored = []
        for n, label in candidates.items():
            tscore = text_match_score(label, n, query_words)

            # If it is neither a strong match nor text-matching, skip
            if tscore == 0 and n not in strong_map:
                continue

            best_tree_bonus = 0
            best_prefix = 0
            for sn in seed_notations:
                d = tree_distance(n, sn)
                best_tree_bonus = max(best_tree_bonus, max(20 - d * 5, 0))
                best_prefix = max(best_prefix, prefix_bonus(sn, n))

            final_score = tscore + best_tree_bonus + best_prefix
            scored.append((final_score, n, label))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:limit]

if __name__ == "__main__":
    query = input("Enter your Iconclass search query: ").strip()
    print("Loading results...")
    start = time.time() 
    results = search_iconclass_classic_sqlite(query, lang="en", limit=100)
    for score, notation, label in results:
        print(f"{score:3d}  {notation} → {label}")
    print(time.time() - start)