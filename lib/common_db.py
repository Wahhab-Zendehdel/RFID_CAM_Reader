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
    cur = conn.cursor()
    cur.execute("SELECT 1")


def _parse_mysql_connection_string(conn_str: str) -> Dict[str, Any]:
    parsed = urlparse(conn_str)
    if parsed.scheme not in ("mysql", "mysql+pymysql"):
        raise ValueError("Unsupported DB scheme: %s" % parsed.scheme)
    user = parsed.username or ""
    password = parsed.password or ""
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 3306
    db = (parsed.path or "").lstrip("/")
    return {"user": user, "password": password, "host": host, "port": port, "db": db}


def _to_mysql_datetime(dt_val: Any) -> Any:
    if isinstance(dt_val, str) and dt_val:
        try:
            dt = datetime.fromisoformat(dt_val.rstrip("Z"))
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return dt_val
    return dt_val


def init_db(db_cfg: Dict[str, Any]) -> None:
    """Ensure local sqlite table and optionally verify MySQL connectivity."""
    mysql_cfg = (db_cfg or {}).get("mysql") or {}
    path = str(db_cfg.get("path") or "data/app.db")
    _ensure_sqlite_table(path)

    if not mysql_cfg.get("enabled"):
        return

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
        print(f"Warning: unable to verify MySQL connectivity: {exc}; MySQL inserts will be retried at runtime")


def store_result(payload: Dict[str, Any], db_cfg: Dict[str, Any]) -> Optional[int]:
    """Insert payload into SQLite and optionally into MySQL.

    Returns sqlite row id when local insert succeeds, else None.
    """
    mysql_cfg = (db_cfg or {}).get("mysql") or {}
    tags_json = json.dumps(payload.get("tags") or [])
    errors_val = payload.get("errors")
    errors_json = json.dumps(errors_val) if errors_val is not None else None
    sqlite_row_id: Optional[int] = None

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
            sqlite_row_id = int(cur.lastrowid)
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:
        print(f"SQLite insert failed: {exc}")

    if not mysql_cfg.get("enabled"):
        return sqlite_row_id

    conn_str = str(mysql_cfg.get("connection_string") or "").strip()
    if not conn_str:
        print("MySQL enabled but no connection_string provided; skipping MySQL insert")
        return sqlite_row_id
    params = _parse_mysql_connection_string(conn_str)
    try:
        import pymysql
    except Exception:
        print("pymysql not installed; skipping MySQL insert")
        return sqlite_row_id

    last_exc: Optional[BaseException] = None
    mysql_row_id: Optional[int] = None
    for _ in range(2):
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
                mysql_dt = _to_mysql_datetime(payload.get("datetime"))
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
                mysql_row_id = int(cur.lastrowid) if cur.lastrowid is not None else None
                last_exc = None
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

    if sqlite_row_id is None:
        return None

    if mysql_row_id is not None:
        try:
            conn = sqlite3.connect(db_path, timeout=30)
            try:
                cur = conn.cursor()
                cur.execute(
                    "UPDATE records SET synced = 1, mysql_id = ?, last_error = NULL WHERE id = ?",
                    (mysql_row_id, sqlite_row_id),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            pass
    elif last_exc is not None:
        print(f"MySQL insert failed after retries: {last_exc}; continuing with local storage only")
        try:
            conn = sqlite3.connect(db_path, timeout=30)
            try:
                cur = conn.cursor()
                cur.execute(
                    "UPDATE records SET synced = 0, last_error = ? WHERE id = ?",
                    (str(last_exc), sqlite_row_id),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            pass

    return sqlite_row_id


def update_result(record_id: int, payload: Dict[str, Any], db_cfg: Dict[str, Any]) -> bool:
    """Update an existing row in SQLite and optionally in MySQL.

    Returns False only when local sqlite update fails or row does not exist.
    """
    mysql_cfg = (db_cfg or {}).get("mysql") or {}
    tags_json = json.dumps(payload.get("tags") or [])
    errors_val = payload.get("errors")
    errors_json = json.dumps(errors_val) if errors_val is not None else None
    db_path = str(db_cfg.get("path") or "data/app.db")

    try:
        _ensure_sqlite_table(db_path)
    except Exception:
        pass

    mysql_id: Optional[int] = None
    try:
        conn = sqlite3.connect(db_path, timeout=30)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE records
                SET
                    datetime = ?,
                    station = ?,
                    station_id = ?,
                    station_type = ?,
                    tags = ?,
                    number = ?,
                    primary_image_url = ?,
                    secondary_image_url = ?,
                    label_image_url = ?,
                    rfid_device_id = ?,
                    primary_cam_id = ?,
                    secondary_cam_id = ?,
                    errors = ?,
                    synced = 0,
                    last_error = NULL
                WHERE id = ?
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
                    record_id,
                ),
            )
            if cur.rowcount <= 0:
                conn.commit()
                return False
            cur.execute("SELECT mysql_id FROM records WHERE id = ?", (record_id,))
            row = cur.fetchone()
            if row:
                mysql_id = row[0]
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:
        print(f"SQLite update failed: {exc}")
        return False

    if not mysql_cfg.get("enabled") or mysql_id is None:
        return True

    conn_str = str(mysql_cfg.get("connection_string") or "").strip()
    if not conn_str:
        return True
    params = _parse_mysql_connection_string(conn_str)
    try:
        import pymysql
    except Exception:
        return True

    mysql_updated = False
    mysql_err = ""
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
            mysql_dt = _to_mysql_datetime(payload.get("datetime"))
            cur.execute(
                """
                UPDATE records
                SET
                    datetime = %s,
                    station = %s,
                    station_id = %s,
                    station_type = %s,
                    tags = %s,
                    number = %s,
                    primary_image_url = %s,
                    secondary_image_url = %s,
                    label_image_url = %s,
                    rfid_device_id = %s,
                    primary_cam_id = %s,
                    secondary_cam_id = %s,
                    errors = %s
                WHERE id = %s
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
                    int(mysql_id),
                ),
            )
            conn.commit()
            mysql_updated = True
        finally:
            try:
                conn.close()
            except Exception:
                pass
    except Exception as exc:
        mysql_err = str(exc)

    if mysql_updated:
        try:
            conn = sqlite3.connect(db_path, timeout=30)
            try:
                cur = conn.cursor()
                cur.execute("UPDATE records SET synced = 1, last_error = NULL WHERE id = ?", (record_id,))
                conn.commit()
            finally:
                conn.close()
        except Exception:
            pass
    elif mysql_err:
        try:
            conn = sqlite3.connect(db_path, timeout=30)
            try:
                cur = conn.cursor()
                cur.execute("UPDATE records SET synced = 0, last_error = ? WHERE id = ?", (mysql_err, record_id))
                conn.commit()
            finally:
                conn.close()
        except Exception:
            pass

    return True
