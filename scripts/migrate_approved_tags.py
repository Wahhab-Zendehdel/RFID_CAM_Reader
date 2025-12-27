import json
from pathlib import Path
from typing import List, Tuple

from core.config import resolve_path
from core.utils import normalize_label_value, normalize_tag
from db.database import init_db
from db.repository import VehicleRepository


def load_approved_tags(path: Path) -> List[Tuple[str, str]]:
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    tags = {}

    def add_tag_entry(raw_tag, raw_label=""):
        if not isinstance(raw_tag, str):
            return
        normalized = normalize_tag(raw_tag)
        if not normalized:
            return
        tags[normalized] = normalize_label_value(raw_label)

    def ingest_list(raw_list):
        for item in raw_list:
            if isinstance(item, str):
                add_tag_entry(item, "")
            elif isinstance(item, dict):
                tag_val = item.get("tag")
                label_val = item.get("label", "")
                if tag_val:
                    add_tag_entry(tag_val, label_val)

    def ingest_mapping(raw_map):
        for raw_tag, info in raw_map.items():
            if isinstance(info, dict):
                label_val = info.get("label", "")
            else:
                label_val = info
            add_tag_entry(raw_tag, label_val)

    if isinstance(data, list):
        ingest_list(data)
    elif isinstance(data, dict):
        raw_approved = data.get("approved_tags")
        if isinstance(raw_approved, dict):
            ingest_mapping(raw_approved)
        elif isinstance(raw_approved, list):
            ingest_list(raw_approved)

        raw_tags = data.get("tags")
        if isinstance(raw_tags, list):
            ingest_list(raw_tags)

        if not isinstance(raw_approved, (dict, list)) and not isinstance(raw_tags, list):
            ingest_mapping(data)

    return list(tags.items())


def main():
    base_dir = Path(__file__).resolve().parent.parent
    approved_path = base_dir / "approved_tags.json"

    db_path = resolve_path("data/app.db")
    init_db(db_path)
    repo = VehicleRepository(db_path)

    entries = load_approved_tags(approved_path)
    if not entries:
        print("No approved tags found to migrate.")
        return

    migrated = 0
    for tag, label in entries:
        existing = repo.find_vehicle_by_tag(tag)
        if existing:
            if label:
                repo.upsert_vehicle({
                    "id": existing.get("id"),
                    "tags": existing.get("tags") or [],
                    "labels": [label],
                })
            continue
        repo.upsert_vehicle({
            "tags": [tag],
            "labels": [label] if label else [],
        })
        migrated += 1

    print(f"Migrated {migrated} tags into vehicles.")


if __name__ == "__main__":
    main()
