import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.common_config import load_config
from lib.common_db import init_db, store_result
from datetime import datetime


def main():
    cfg = load_config()
    db_cfg = cfg.get("db") or {}
    print("Initializing DB (will create table if needed)...")
    init_db(db_cfg)

    payload = {
        "datetime": datetime.utcnow().isoformat() + "Z",
        "station": "test_station",
        "tags": ["TAG001"],
        "number": "12345",
        "primary_image_url": "http://example.local/images/1.jpg",
        "secondary_image_url": None,
        "label_image_url": None,
        "message": "test insert",
        "errors": None,
    }

    print("Storing test payload...")
    try:
        store_result(payload, db_cfg)
        print("✓ Insert succeeded")
    except Exception as e:
        print(f"✗ Insert failed: {e}")


if __name__ == "__main__":
    main()
