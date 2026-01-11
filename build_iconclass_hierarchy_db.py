import os
import sqlite3
from iconclass import init

DB_PATH = os.path.join("Evaluation-of-Text-Representation-Methods-for-ICONCLASS-Label-Recommendation", "iconclass_hierarchy.db")

LANGS = ["en"]

def walk(node):
    """Yield node and all descendants recursively."""
    yield node
    for child in node:
        yield from walk(child)


def notation_of(node):
    """Extract notation code from iconclass node (as in your original code)."""
    return repr(node).split()[0]


def build_db():
    ic = init()

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # Normal table: includes hierarchy info
    cur.execute("""
        CREATE TABLE IF NOT EXISTS iconclass (
            notation TEXT NOT NULL,
            lang     TEXT NOT NULL,
            label    TEXT NOT NULL,
            parent   TEXT,            -- parent notation (NULL for root)
            depth    INTEGER NOT NULL, -- depth in tree (root = 0)
            PRIMARY KEY (notation, lang)
        );
    """)

    # Useful for hierarchy queries
    cur.execute("CREATE INDEX IF NOT EXISTS idx_iconclass_parent ON iconclass(parent);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_iconclass_depth  ON iconclass(depth);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_iconclass_notation ON iconclass(notation);")

    # FTS5 table
    cur.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS iconclass_fts
        USING fts5(notation, lang, label);
    """)

    # For repeatable "build"
    cur.execute("DELETE FROM iconclass;")
    cur.execute("DELETE FROM iconclass_fts;")
    con.commit()

    inserted = 0
    skipped_empty = 0

    for node in walk(ic):
        notation = notation_of(node)

        path_nodes = list(node.path())
        depth = len(path_nodes) - 1

        parent = None
        if len(path_nodes) >= 2:
            parent = notation_of(path_nodes[-2])

        for lang in LANGS:
            try:
                label = node(lang) or ""
            except Exception:
                label = ""

            # Note for self: find these in the data
            label = label.strip()
            if not label:
                skipped_empty += 1
                continue

            # store in normal table
            cur.execute(
                "INSERT OR REPLACE INTO iconclass(notation, lang, label, parent, depth) VALUES (?, ?, ?, ?, ?);",
                (notation, lang, label, parent, depth)
            )

            # store in FTS index (position-based insert for FTS)
            cur.execute(
                "INSERT INTO iconclass_fts VALUES (?, ?, ?);",
                (notation, lang, label)
            )

            inserted += 1

            # To see progress is being made:
            if inserted % 100000 == 0:
                con.commit()
                print(f"{inserted} entries...")

    con.commit()
    con.close()

    print(f"Database built: iconclass_hierarchy.db — {inserted} entries.")
    print(f"Empty notations: {skipped_empty}")


if __name__ == "__main__":
    build_db()