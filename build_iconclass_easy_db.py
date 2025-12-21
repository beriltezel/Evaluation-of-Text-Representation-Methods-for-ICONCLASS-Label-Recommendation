import os
import sqlite3
from iconclass import init

DB_PATH = os.path.join("Evaluation-of-Text-Representation-Methods-for-ICONCLASS-Label-Recommendation", "iconclass_easy.db")
LANGS = ["en"]

def walk(node):
    yield node
    for child in node:
        yield from walk(child)

def notation_of(node):
    return repr(node).split()[0]

def build_db():
    ic = init()

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # normal table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS iconclass (
            notation TEXT NOT NULL,
            lang     TEXT NOT NULL,
            label    TEXT NOT NULL,
            PRIMARY KEY (notation, lang)
        );
    """)

    # FTS5 index
    cur.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS iconclass_fts
        USING fts5(notation, lang, label);
    """)

    # for repeatable "build"
    cur.execute("DELETE FROM iconclass;")
    cur.execute("DELETE FROM iconclass_fts;")
    con.commit()

    inserted = 0
    for node in walk(ic):
        notation = notation_of(node)

        for lang in LANGS:
            try:
                label = node(lang) or ""
            except Exception:
                label = ""

            label = label.strip()
            if not label:
                continue

            cur.execute(
                "INSERT OR REPLACE INTO iconclass(notation, lang, label) VALUES (?, ?, ?);",
                (notation, lang, label)
            )
            cur.execute(
                "INSERT INTO iconclass_fts VALUES (?, ?, ?);",
                (notation, lang, label)
            )
            inserted += 1

        if inserted % 5000 == 0 and inserted > 0:
            con.commit()

    con.commit()
    con.close()
    print(f"Databank is built: iconclass_easy.db — {inserted} entries (Notation + Language).")

if __name__ == "__main__":
    build_db()