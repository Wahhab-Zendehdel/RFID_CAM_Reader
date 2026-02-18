from typing import Any, Dict, Optional
import sqlite3
from pathlib import Path
import json
from urllib.parse import urlparse
from datetime import datetime


def _ensure_sqlite_table(db_path: str) -> None:
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                datetime TEXT,
                station TEXT,
                station_id INTEGER,
                station_type TEXT,
                tags TEXT,
                number TEXT,
                primary_image_url TEXT,
                secondary_image_url TEXT,
                label_image_url TEXT,
                rfid_device_id INTEGER,
                primary_cam_id INTEGER,
                secondary_cam_id INTEGER,
                errors TEXT
            )
            """
        )

        # Ensure additional columns exist (migration for older DBs)
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(records)")
        existing = {row[1] for row in cur.fetchall()}
        extras = [
            ("station_id", "INTEGER", ""),
            ("station_type", "TEXT", ""),
            ("rfid_device_id", "INTEGER", ""),
            ("primary_cam_id", "INTEGER", ""),
            ("secondary_cam_id", "INTEGER", ""),
            ("created_at", "TEXT", ""),
            ("synced", "INTEGER", ""),
            ("mysql_id", "INTEGER", ""),
            ("last_error", "TEXT", ""),
        ]
        for name, typ, default in extras:
            if name not in existing:
                sql = f"ALTER TABLE records ADD COLUMN {name} {typ} {default}".strip()
                cur.execute(sql)
        # Set reasonable values for existing rows after adding columns
        if "created_at" not in existing:
            try:
                cur.execute("UPDATE records SET created_at = COALESCE(datetime, CURRENT_TIMESTAMP) WHERE created_at IS NULL")
            except Exception:
                pass
        if "synced" not in existing:
            try:
                cur.execute("UPDATE records SET synced = 0 WHERE synced IS NULL")
            except Exception:
                pass
        conn.commit()
    finally:
        conn.close()


def _ensure_mysql_table(conn) -> None:
    # Assume MySQL table `records` is created beforehand per user request.
    # This function intentionally does not create the table.
    # We still exercise a simple no-op cursor call to validate the connection.
    cur = conn.cursor()
    cur.execute("SELECT 1")
    # do not commit anything


def _parse_mysql_connection_string(conn_str: str) -> Dict[str, Any]:
    # Accept formats like mysql://user:pass@host:port/dbname
    parsed = urlparse(conn_str)
    if parsed.scheme not in ("mysql", "mysql+pymysql"):
        raise ValueError("Unsupported DB scheme: %s" % parsed.scheme)
    user = parsed.username or ""
    password = parsed.password or ""
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 3306
    db = (parsed.path or "").lstrip("/")
    return {"user": user, "password": password, "host": host, "port": port, "db": db}


def init_db(db_cfg: Dict[str, Any]) -> None:
    """Ensure the results table exists.

    Always create/ensure the local SQLite table. If MySQL is enabled, verify
    connectivity but do not rely on it for the canonical storage.
    """
    mysql_cfg = (db_cfg or {}).get("mysql") or {}
    # Ensure local sqlite table exists for offline/local copies.
    path = str(db_cfg.get("path") or "data/app.db")
    _ensure_sqlite_table(path)

    # Optionally verify MySQL connectivity for best-effort remote insertion.
    if mysql_cfg.get("enabled"):
        conn_str = str(mysql_cfg.get("connection_string") or "").strip()
        if not conn_str:
            raise ValueError("MySQL enabled but no connection_string provided")
        params = _parse_mysql_connection_string(conn_str)
        try:
            import pymysql
        except Exception as exc:
            raise RuntimeError("pymysql is required for MySQL support; install with 'pip install pymysql'") from exc
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
                _ensure_mysql_table(conn)
            finally:
                conn.close()
        except Exception as exc:
            print(f"✗ Warning: unable to verify MySQL connectivity: {exc}; MySQL inserts will be retried at runtime")


def store_result(payload: Dict[str, Any], db_cfg: Dict[str, Any]) -> None:
    """Insert payload into both SQLite and MySQL (if enabled).

    Always writes to local SQLite. If MySQL is enabled, attempt to also write
    to MySQL; failures there are logged but do not prevent the local insert.
    """
    mysql_cfg = (db_cfg or {}).get("mysql") or {}
    tags_json = json.dumps(payload.get("tags") or [])
    errors_val = payload.get("errors")
    errors_json = json.dumps(errors_val) if errors_val is not None else None

    # Ensure local sqlite table exists and write the record there.
    db_path = str(db_cfg.get("path") or "data/app.db")
    try:
        _ensure_sqlite_table(db_path)
    except Exception:
        pass

    try:
        conn = sqlite3.connect(db_path, timeout=30)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO records (
                    datetime, station, station_id, station_type, tags, number,
                    primary_image_url, secondary_image_url, label_image_url,
                    rfid_device_id, primary_cam_id, secondary_cam_id,
                    errors, created_at, synced, mysql_id, last_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload.get("datetime"),
                    payload.get("station"),
                    payload.get("station_id"),
                    payload.get("station_type"),
                    tags_json,
                    payload.get("number"),
                    payload.get("primary_image_url"),
                    payload.get("secondary_image_url"),
                    payload.get("label_image_url"),
                    payload.get("rfid_device_id"),
                    payload.get("primary_cam_id"),
                    payload.get("secondary_cam_id"),
                    errors_json,
                    payload.get("datetime") or datetime.utcnow().isoformat(),
                    0,
                    None,
                    None,
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:
        print(f"✗ SQLite insert failed: {exc}")

    # If MySQL enabled, attempt to insert there as well (best-effort).
    if not mysql_cfg.get("enabled"):
        return

    conn_str = str(mysql_cfg.get("connection_string") or "").strip()
    if not conn_str:
        print("✗ MySQL enabled but no connection_string provided; skipping MySQL insert")
        return
    params = _parse_mysql_connection_string(conn_str)
    try:
        import pymysql
    except Exception:
        print("✗ pymysql not installed; skipping MySQL insert")
        return

    last_exc: Optional[BaseException] = None
    for attempt in range(2):
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
                dt_val = payload.get("datetime")
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
                        mysql_dt if mysql_dt is not None else payload.get("datetime"),
                        payload.get("station"),
                        payload.get("station_id"),
                        payload.get("station_type"),
                        tags_json,
                        payload.get("number"),
                        payload.get("primary_image_url"),
                        payload.get("secondary_image_url"),
                        payload.get("label_image_url"),
                        payload.get("rfid_device_id"),
                        payload.get("primary_cam_id"),
                        payload.get("secondary_cam_id"),
                        errors_json,
                    ),
                )
                conn.commit()
                break
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        except Exception as exc:
            last_exc = exc
            try:
                import time

                time.sleep(1)
            except Exception:
                pass

    if last_exc is not None:
        print(f"✗ MySQL insert failed after retries: {last_exc}; continuing with local storage only")
