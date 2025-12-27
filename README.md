Run: `python app.py`
Open: `http://127.0.0.1:5000`

Notes:
- SQLite DB at `data/app.db` is created on startup.
- Logs are written to `logs/process.jsonl`.
- Images captured for vehicles are stored under `data/images`.
- Configurable vehicle slots via `config.json`:
  `vehicles.default_tag_slots` and `vehicles.default_label_slots`.
- Optional migration from `approved_tags.json`:
  `python scripts/migrate_approved_tags.py`
- Schema migration from legacy vehicles table:
  `python scripts/migrate_vehicle_schema.py`
