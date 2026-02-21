from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from lib.common_config import load_config, load_env_file
from lib.common_db import _ensure_sqlite_table


ALLOWED_SORT_COLUMNS = {
    "id",
    "datetime",
    "created_at",
    "station",
    "number",
    "station_type",
}


def _model_dump(model: Any, exclude_unset: bool = False) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_unset=exclude_unset)
    return model.dict(exclude_unset=exclude_unset)


def _loads_or_default(raw: Optional[str], default: Any) -> Any:
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default


def _resolve_db_path() -> str:
    load_env_file()
    cfg = load_config()
    db_cfg = cfg.get("db") or {}
    return str(db_cfg.get("path") or "data/app.db")


class RecordStore:
    def __init__(self):
        self.db_path = _resolve_db_path()
        _ensure_sqlite_table(self.db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "datetime": row["datetime"],
            "station": row["station"],
            "station_id": row["station_id"],
            "station_type": row["station_type"],
            "tags": _loads_or_default(row["tags"], []),
            "number": row["number"],
            "primary_image_url": row["primary_image_url"],
            "secondary_image_url": row["secondary_image_url"],
            "label_image_url": row["label_image_url"],
            "rfid_device_id": row["rfid_device_id"],
            "primary_cam_id": row["primary_cam_id"],
            "secondary_cam_id": row["secondary_cam_id"],
            "errors": _loads_or_default(row["errors"], None),
            "created_at": row["created_at"],
            "synced": row["synced"],
            "mysql_id": row["mysql_id"],
            "last_error": row["last_error"],
        }

    def create_record(self, payload_model: Any) -> int:
        payload = _model_dump(payload_model)
        dt_val = payload.get("datetime") or datetime.utcnow().isoformat()
        created_at = datetime.utcnow().isoformat()
        tags_json = json.dumps(payload.get("tags") or [])
        errors_json = json.dumps(payload.get("errors")) if payload.get("errors") is not None else None

        conn = self._connect()
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
                    dt_val,
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
                    created_at,
                    0,
                    None,
                    None,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()

    def create_records_bulk(self, payload_models: Sequence[Any]) -> List[int]:
        inserted_ids: List[int] = []
        conn = self._connect()
        try:
            cur = conn.cursor()
            for payload_model in payload_models:
                payload = _model_dump(payload_model)
                dt_val = payload.get("datetime") or datetime.utcnow().isoformat()
                created_at = datetime.utcnow().isoformat()
                tags_json = json.dumps(payload.get("tags") or [])
                errors_json = json.dumps(payload.get("errors")) if payload.get("errors") is not None else None
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
                        dt_val,
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
                        created_at,
                        0,
                        None,
                        None,
                    ),
                )
                inserted_ids.append(int(cur.lastrowid))
            conn.commit()
            return inserted_ids
        finally:
            conn.close()

    def get_record(self, record_id: int) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM records WHERE id = ?", (record_id,))
            row = cur.fetchone()
            if row is None:
                return None
            return self._row_to_dict(row)
        finally:
            conn.close()

    def list_records(
        self,
        station: Optional[str],
        station_id: Optional[int],
        station_type: Optional[str],
        tag: Optional[str],
        number: Optional[str],
        has_errors: Optional[bool],
        search: Optional[str],
        date_from: Optional[str],
        date_to: Optional[str],
        page: int,
        page_size: int,
        sort_by: str,
        sort_dir: str,
    ) -> Tuple[List[Dict[str, Any]], int]:
        where_parts: List[str] = []
        params: List[Any] = []

        if station:
            where_parts.append("station = ?")
            params.append(station)
        if station_id is not None:
            where_parts.append("station_id = ?")
            params.append(station_id)
        if station_type:
            where_parts.append("station_type = ?")
            params.append(station_type)
        if tag:
            where_parts.append("tags LIKE ?")
            params.append('%"' + tag + '"%')
        if number:
            where_parts.append("number LIKE ?")
            params.append("%" + number + "%")
        if has_errors is True:
            where_parts.append("errors IS NOT NULL AND errors <> '[]' AND errors <> ''")
        elif has_errors is False:
            where_parts.append("(errors IS NULL OR errors = '[]' OR errors = '')")
        if search:
            where_parts.append("(station LIKE ? OR number LIKE ? OR tags LIKE ? OR station_type LIKE ?)")
            search_val = "%" + search + "%"
            params.extend([search_val, search_val, search_val, search_val])
        if date_from:
            where_parts.append("datetime >= ?")
            params.append(date_from)
        if date_to:
            where_parts.append("datetime <= ?")
            params.append(date_to)

        where_sql = ""
        if where_parts:
            where_sql = " WHERE " + " AND ".join(where_parts)

        order_column = sort_by if sort_by in ALLOWED_SORT_COLUMNS else "id"
        order_dir = "DESC" if str(sort_dir).lower() == "desc" else "ASC"
        offset = (page - 1) * page_size

        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(1) AS total FROM records" + where_sql, tuple(params))
            total = int(cur.fetchone()["total"])

            query = (
                "SELECT * FROM records"
                + where_sql
                + " ORDER BY "
                + order_column
                + " "
                + order_dir
                + " LIMIT ? OFFSET ?"
            )
            query_params = list(params) + [page_size, offset]
            cur.execute(query, tuple(query_params))
            rows = cur.fetchall()
            return [self._row_to_dict(r) for r in rows], total
        finally:
            conn.close()

    def update_record(self, record_id: int, patch_model: Any) -> Optional[Dict[str, Any]]:
        patch = _model_dump(patch_model, exclude_unset=True)
        if not patch:
            return self.get_record(record_id)

        assignments: List[str] = []
        params: List[Any] = []

        for key in [
            "datetime",
            "station",
            "station_id",
            "station_type",
            "number",
            "primary_image_url",
            "secondary_image_url",
            "label_image_url",
            "rfid_device_id",
            "primary_cam_id",
            "secondary_cam_id",
            "created_at",
            "synced",
            "mysql_id",
            "last_error",
        ]:
            if key in patch:
                assignments.append(key + " = ?")
                params.append(patch.get(key))

        if "tags" in patch:
            assignments.append("tags = ?")
            params.append(json.dumps(patch.get("tags") or []))
        if "errors" in patch:
            errors_val = patch.get("errors")
            params.append(json.dumps(errors_val) if errors_val is not None else None)
            assignments.append("errors = ?")

        if not assignments:
            return self.get_record(record_id)

        params.append(record_id)
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute("UPDATE records SET " + ", ".join(assignments) + " WHERE id = ?", tuple(params))
            conn.commit()
            if cur.rowcount <= 0:
                return None
            return self.get_record(record_id)
        finally:
            conn.close()

    def delete_record(self, record_id: int) -> bool:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM records WHERE id = ?", (record_id,))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def get_stats(
        self,
        station: Optional[str],
        station_id: Optional[int],
        date_from: Optional[str],
        date_to: Optional[str],
    ) -> Dict[str, Any]:
        where_parts: List[str] = []
        params: List[Any] = []
        if station:
            where_parts.append("station = ?")
            params.append(station)
        if station_id is not None:
            where_parts.append("station_id = ?")
            params.append(station_id)
        if date_from:
            where_parts.append("datetime >= ?")
            params.append(date_from)
        if date_to:
            where_parts.append("datetime <= ?")
            params.append(date_to)

        where_sql = ""
        if where_parts:
            where_sql = " WHERE " + " AND ".join(where_parts)

        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(1) AS total FROM records" + where_sql, tuple(params))
            total_records = int(cur.fetchone()["total"])

            cur.execute(
                "SELECT COUNT(1) AS total FROM records"
                + where_sql
                + (" AND " if where_sql else " WHERE ")
                + "errors IS NOT NULL AND errors <> '[]' AND errors <> ''",
                tuple(params),
            )
            records_with_errors = int(cur.fetchone()["total"])

            cur.execute(
                "SELECT COUNT(1) AS total FROM records"
                + where_sql
                + (" AND " if where_sql else " WHERE ")
                + "number IS NOT NULL AND number <> ''",
                tuple(params),
            )
            records_with_number = int(cur.fetchone()["total"])

            cur.execute("SELECT MIN(datetime) AS date_min, MAX(datetime) AS date_max FROM records" + where_sql, tuple(params))
            row = cur.fetchone()
            date_min = row["date_min"]
            date_max = row["date_max"]

            cur.execute(
                "SELECT station, COUNT(1) AS count FROM records"
                + where_sql
                + " GROUP BY station ORDER BY count DESC"
            , tuple(params))
            by_station = [{"station": r["station"], "count": int(r["count"])} for r in cur.fetchall()]

            cur.execute(
                "SELECT station_type AS station, COUNT(1) AS count FROM records"
                + where_sql
                + " GROUP BY station_type ORDER BY count DESC"
            , tuple(params))
            by_station_type = [{"station": r["station"], "count": int(r["count"])} for r in cur.fetchall()]

            return {
                "total_records": total_records,
                "records_with_errors": records_with_errors,
                "records_with_number": records_with_number,
                "date_min": date_min,
                "date_max": date_max,
                "by_station": by_station,
                "by_station_type": by_station_type,
            }
        finally:
            conn.close()
