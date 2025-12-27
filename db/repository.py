import json
import sqlite3
from typing import List, Optional

from db.database import get_connection
from core.utils import normalize_label_value, normalize_tag


class VehicleRepository:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._json1_available = None

    def _parse_json_list(self, raw_value: str) -> list:
        if not raw_value:
            return []
        try:
            data = json.loads(raw_value)
        except Exception:
            return []
        if not isinstance(data, list):
            return []
        return [str(item) for item in data if str(item).strip()]

    def _json1_supported(self) -> bool:
        if self._json1_available is not None:
            return self._json1_available
        try:
            with get_connection(self.db_path) as conn:
                conn.execute("SELECT json('[]');")
            self._json1_available = True
        except sqlite3.Error:
            self._json1_available = False
        return self._json1_available

    def list_vehicles(self) -> list:
        with get_connection(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM vehicles ORDER BY id ASC;").fetchall()
        items = []
        for row in rows:
            payload = dict(row)
            payload["tags"] = self._parse_json_list(payload.get("tags_json"))
            payload["labels"] = self._parse_json_list(payload.get("labels_json"))
            payload.pop("tags_json", None)
            payload.pop("labels_json", None)
            items.append(payload)
        return items

    def get_vehicle_by_id(self, vehicle_id: int):
        with get_connection(self.db_path) as conn:
            row = conn.execute("SELECT * FROM vehicles WHERE id = ?;", (vehicle_id,)).fetchone()
        if not row:
            return None
        payload = dict(row)
        payload["tags"] = self._parse_json_list(payload.get("tags_json"))
        payload["labels"] = self._parse_json_list(payload.get("labels_json"))
        payload.pop("tags_json", None)
        payload.pop("labels_json", None)
        return payload

    def find_vehicle_by_tag(self, tag: str):
        norm = normalize_tag(tag)
        if not norm:
            return None
        if self._json1_supported():
            with get_connection(self.db_path) as conn:
                row = conn.execute(
                    """
                    SELECT * FROM vehicles
                    WHERE EXISTS (
                        SELECT 1 FROM json_each(vehicles.tags_json)
                        WHERE value = ?
                    )
                    LIMIT 1;
                    """,
                    (norm,),
                ).fetchone()
            if not row:
                return None
            payload = dict(row)
            payload["tags"] = self._parse_json_list(payload.get("tags_json"))
            payload["labels"] = self._parse_json_list(payload.get("labels_json"))
            payload.pop("tags_json", None)
            payload.pop("labels_json", None)
            return payload

        vehicles = self.list_vehicles()
        for vehicle in vehicles:
            if norm in vehicle.get("tags", []):
                return vehicle
        return None

    def find_vehicle_by_any_tag(self, tags: List[str]):
        tags = [normalize_tag(tag) for tag in (tags or []) if tag]
        tags = [tag for tag in tags if tag]
        if not tags:
            return None
        if self._json1_supported():
            placeholders = ",".join("?" for _ in tags)
            query = f"""
                SELECT * FROM vehicles
                WHERE EXISTS (
                    SELECT 1 FROM json_each(vehicles.tags_json)
                    WHERE value IN ({placeholders})
                )
                LIMIT 1;
            """
            with get_connection(self.db_path) as conn:
                row = conn.execute(query, tags).fetchone()
            if not row:
                return None
            payload = dict(row)
            payload["tags"] = self._parse_json_list(payload.get("tags_json"))
            payload["labels"] = self._parse_json_list(payload.get("labels_json"))
            payload.pop("tags_json", None)
            payload.pop("labels_json", None)
            return payload

        vehicles = self.list_vehicles()
        for vehicle in vehicles:
            vehicle_tags = vehicle.get("tags", [])
            if any(tag in vehicle_tags for tag in tags):
                return vehicle
        return None

    def upsert_vehicle(self, payload: dict) -> dict:
        vehicle_id = payload.get("id")
        tags = payload.get("tags")
        if tags is None:
            tags = [payload.get("tag1"), payload.get("tag2"), payload.get("tag3"), payload.get("tag4"), payload.get("tag5")]
        tags = tags or []
        normalized = []
        for tag in tags:
            norm = normalize_tag(tag)
            if norm:
                normalized.append(norm)
        unique_tags = []
        for tag in normalized:
            if tag not in unique_tags:
                unique_tags.append(tag)
        while len(unique_tags) < 5:
            unique_tags.append("")
        unique_tags = unique_tags[:5]

        labels = payload.get("labels")
        if labels is None:
            labels = [payload.get("label")] if payload.get("label") is not None else []
        labels = labels or []
        normalized_labels = []
        for label in labels:
            norm_label = normalize_label_value(label)
            if norm_label:
                normalized_labels.append(norm_label)
        unique_labels = []
        for label in normalized_labels:
            if label not in unique_labels:
                unique_labels.append(label)

        with get_connection(self.db_path) as conn:
            if vehicle_id:
                existing = conn.execute("SELECT id FROM vehicles WHERE id = ?;", (vehicle_id,)).fetchone()
            else:
                existing = None

            if existing:
                conn.execute(
                    """
                    UPDATE vehicles
                    SET tags_json = ?, labels_json = ?
                    WHERE id = ?;
                    """,
                    (json.dumps(unique_tags), json.dumps(unique_labels), vehicle_id),
                )
                row = conn.execute("SELECT * FROM vehicles WHERE id = ?;", (vehicle_id,)).fetchone()
            else:
                conn.execute(
                    """
                    INSERT INTO vehicles (tags_json, labels_json)
                    VALUES (?, ?);
                    """,
                    (json.dumps(unique_tags), json.dumps(unique_labels)),
                )
                new_id = conn.execute("SELECT last_insert_rowid();").fetchone()[0]
                row = conn.execute("SELECT * FROM vehicles WHERE id = ?;", (new_id,)).fetchone()

        if not row:
            return None
        payload = dict(row)
        payload["tags"] = self._parse_json_list(payload.get("tags_json"))
        payload["labels"] = self._parse_json_list(payload.get("labels_json"))
        payload.pop("tags_json", None)
        payload.pop("labels_json", None)
        return payload

    def delete_vehicle(self, vehicle_id: int) -> bool:
        with get_connection(self.db_path) as conn:
            conn.execute("DELETE FROM vehicle_images WHERE id = ?;", (vehicle_id,))
            cur = conn.execute("DELETE FROM vehicles WHERE id = ?;", (vehicle_id,))
            return cur.rowcount > 0

    def get_vehicle_images(self, vehicle_id: int) -> Optional[dict]:
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM vehicle_images WHERE id = ?;", (vehicle_id,)
            ).fetchone()
        return dict(row) if row else None

    def upsert_vehicle_images(self, vehicle_id: int, primary_path: Optional[str], secondary_path: Optional[str]) -> None:
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO vehicle_images (id, primary_camera_image, secondary_camera_image)
                VALUES (?, ?, ?);
                """,
                (vehicle_id, primary_path or "", secondary_path or ""),
            )
