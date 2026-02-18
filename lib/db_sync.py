"""Background sync utility: push unsynced sqlite rows to MySQL.

Usage:
    python3 -m lib.db_sync --once
    python3 -m lib.db_sync --interval 30
"""
from typing import Dict, Any, Optional
import sqlite3
import time
import json
from urllib.parse import urlparse
from datetime import datetime


from .common_db import _parse_mysql_connection_string, _ensure_sqlite_table


def _get_sqlite_conn(path: str):
    return sqlite3.connect(path, timeout=30)


def _attempt_mysql_insert(params: Dict[str, Any], row: sqlite3.Row) -> Optional[int]:
    try:
        import pymysql
    except Exception:
        print("✗ pymysql not installed; cannot sync to MySQL")
        return None

    try:
        conn = pymysql.connect(
            host=params["host"],
            user=params["user"],
            password=params["password"],
            port=int(params["port"]),
            database=params["db"],
            autocommit=False,
            connect_timeout=5,
        )
        try:
            cur = conn.cursor()
            dt_val = row["datetime"]
            mysql_dt = None
            if isinstance(dt_val, str) and dt_val:
                try:
                    s = dt_val.rstrip("Z")
                    dt = datetime.fromisoformat(s)
                    mysql_dt = dt.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    mysql_dt = dt_val

            cur.execute(
                """
                INSERT INTO records (
                    datetime, station, station_id, station_type, tags, number,
                    primary_image_url, secondary_image_url, label_image_url,
                    rfid_device_id, primary_cam_id, secondary_cam_id,
                    errors
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    mysql_dt if mysql_dt is not None else row["datetime"],
                    row["station"],
                    row["station_id"],
                    row["station_type"],
                    row["tags"],
                    row["number"],
                    row["primary_image_url"],
                    row["secondary_image_url"],
                    row["label_image_url"],
                    row["rfid_device_id"],
                    row["primary_cam_id"],
                    row["secondary_cam_id"],
                    row["errors"],
                ),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            try:
                conn.close()
            except Exception:
                pass
    except Exception as exc:
        return None


def sync_once(db_cfg: Dict[str, Any]) -> None:
    sqlite_path = str(db_cfg.get("path") or "data/app.db")
    mysql_cfg = (db_cfg or {}).get("mysql") or {}
    if not mysql_cfg.get("enabled"):
        print("MySQL not enabled; nothing to sync")
        return
    conn_str = str(mysql_cfg.get("connection_string") or "").strip()
    if not conn_str:
        print("MySQL connection string missing; aborting sync")
        return
    params = _parse_mysql_connection_string(conn_str)

    # Ensure sqlite table and migrations are applied before reading
    try:
        _ensure_sqlite_table(sqlite_path)
    except Exception:
        pass
    sconn = _get_sqlite_conn(sqlite_path)
    sconn.row_factory = sqlite3.Row
    try:
        cur = sconn.cursor()
        cur.execute("SELECT * FROM records WHERE synced=0 ORDER BY id LIMIT 200")
        rows = cur.fetchall()
        if not rows:
            print("No unsynced rows")
            return
        for r in rows:
            last_err = None
            mysql_id = None
            for attempt in range(2):
                mysql_id = _attempt_mysql_insert(params, r)
                if mysql_id:
                    break
                time.sleep(1)
            if mysql_id:
                try:
                    cur.execute("UPDATE records SET synced=1, mysql_id=?, last_error=NULL WHERE id=?", (mysql_id, r["id"]))
                    sconn.commit()
                    print(f"Synced row {r['id']} -> mysql id {mysql_id}")
                except Exception as e:
                    print(f"Failed to mark row {r['id']} as synced: {e}")
            else:
                try:
                    cur.execute("UPDATE records SET last_error=? WHERE id=?", (str(last_err), r["id"]))
                    sconn.commit()
                except Exception:
                    pass
    finally:
        sconn.close()


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=0, help="poll interval seconds; 0 = once")
    parser.add_argument("--config", type=str, default="config/db.json")
    args = parser.parse_args()

    with open(args.config, "r") as fh:
        cfg = json.load(fh).get("db") or {}

    if args.once or args.interval <= 0:
        sync_once(cfg)
    else:
        while True:
            sync_once(cfg)
            time.sleep(args.interval)
