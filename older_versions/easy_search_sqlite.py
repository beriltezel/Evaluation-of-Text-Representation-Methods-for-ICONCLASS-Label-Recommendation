import os
import sqlite3

DB_PATH = os.path.join("Evaluation-of-Text-Representation-Methods-for-ICONCLASS-Label-Recommendation", "iconclass_easy.db")

def search_iconclass(query, lang="en", limit=50):
    query = query.strip()
    if not query:
        return []

    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row

        # Notation substring
        like = f"%{query.lower()}%"
        rows_code = con.execute("""
            SELECT notation, label
            FROM iconclass
            WHERE lang = ?
              AND lower(notation) LIKE ?
            LIMIT ?;
        """, (lang, like, limit)).fetchall()

        # Label full-text (FTS5)
        rows_fts = con.execute("""
            SELECT notation, label
            FROM iconclass_fts
            WHERE iconclass_fts MATCH ?
              AND lang = ?
            LIMIT ?;
        """, (query, lang, limit)).fetchall()

        # Removing duplicates
        seen = set()
        out = []
        for r in list(rows_code) + list(rows_fts):
            key = (r["notation"], r["label"])
            if key not in seen:
                seen.add(key)
                out.append(r)

        return out[:limit]


if __name__ == "__main__":
    results = search_iconclass("lion", lang="en", limit=50)
    for r in results:
        print(r["notation"], "→", r["label"])