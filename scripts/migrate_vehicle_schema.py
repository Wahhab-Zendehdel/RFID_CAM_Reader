import json
import sqlite3
from pathlib import Path
from typing import List

from core.config import resolve_path
from core.utils import normalize_label_value, normalize_tag
from db.schema import SCHEMA_SQL


def table_columns(conn, table_name: str) -> List[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name});").fetchall()
    return [row[1] for row in rows]


def main():
    db_path = resolve_path("data/app.db")
    if not Path(db_path).exists():
        print(f"Database not found at {db_path}. Nothing to migrate.")
        return

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = OFF;")
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        table_names = {row[0] for row in tables}
        if "vehicles" not in table_names:
            print("No vehicles table found. Nothing to migrate.")
            return

        columns = table_columns(conn, "vehicles")
        if "tags_json" in columns and "labels_json" in columns:
            print("Vehicles table already uses tags_json/labels_json.")
            return

        if not {"tag1", "tag2", "tag3", "tag4", "tag5", "label"}.issubset(columns):
            print("Vehicles table schema not recognized for migration.")
            return

        conn.execute("ALTER TABLE vehicles RENAME TO vehicles_old;")
        conn.executescript(SCHEMA_SQL)

        rows = conn.execute("SELECT id, tag1, tag2, tag3, tag4, tag5, label FROM vehicles_old;").fetchall()
        migrated = 0
        for row in rows:
            vehicle_id, tag1, tag2, tag3, tag4, tag5, label = row
            tags = [normalize_tag(tag) for tag in [tag1, tag2, tag3, tag4, tag5] if tag]
            tags = [tag for tag in tags if tag]
            labels = [normalize_label_value(label)] if label else []
            conn.execute(
                "INSERT INTO vehicles (id, tags_json, labels_json) VALUES (?, ?, ?);",
                (vehicle_id, json.dumps(tags), json.dumps(labels)),
            )
            migrated += 1

        conn.execute("DROP TABLE vehicles_old;")
        conn.commit()
        print(f"Migrated {migrated} vehicles to tags_json/labels_json schema.")
    finally:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.close()


if __name__ == "__main__":
    main()
