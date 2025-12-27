import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from core.utils import normalize_label_value, normalize_tag

from db.schema import SCHEMA_SQL


@contextmanager
def get_connection(db_path: str):
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str) -> None:
    path = Path(db_path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    with get_connection(db_path) as conn:
        _migrate_legacy_schema(conn)
        conn.executescript(SCHEMA_SQL)


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name = ?;",
        (table_name,),
    ).fetchone()
    return bool(row)


def _table_columns(conn: sqlite3.Connection, table_name: str) -> list:
    rows = conn.execute(f"PRAGMA table_info({table_name});").fetchall()
    return [row[1] for row in rows]


def _migrate_legacy_schema(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "vehicles"):
        return

    columns = _table_columns(conn, "vehicles")
    if "tags_json" in columns and "labels_json" in columns:
        return

    legacy_cols = {"tag1", "tag2", "tag3", "tag4", "tag5", "label"}
    if not legacy_cols.issubset(columns):
        return

    conn.execute("PRAGMA foreign_keys = OFF;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS vehicles_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tags_json TEXT NOT NULL,
            labels_json TEXT NOT NULL
        );
        """
    )
    rows = conn.execute(
        "SELECT id, tag1, tag2, tag3, tag4, tag5, label FROM vehicles;"
    ).fetchall()
    for row in rows:
        vehicle_id, tag1, tag2, tag3, tag4, tag5, label = row
        tags = [normalize_tag(tag) for tag in [tag1, tag2, tag3, tag4, tag5] if tag]
        tags = [tag for tag in tags if tag]
        labels = [normalize_label_value(label)] if label else []
        conn.execute(
            "INSERT INTO vehicles_new (id, tags_json, labels_json) VALUES (?, ?, ?);",
            (vehicle_id, json.dumps(tags), json.dumps(labels)),
        )

    conn.execute("DROP TABLE vehicles;")
    conn.execute("ALTER TABLE vehicles_new RENAME TO vehicles;")
    conn.execute("PRAGMA foreign_keys = ON;")
