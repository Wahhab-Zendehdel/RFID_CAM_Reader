from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Optional

from .common_config import load_config
from .common_rfid import RFIDListener
from .models import StationStatus, TagEvent


def _utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


class SangShekanStation:
    """
    Crusher station: listens to RFID and emits normalized tags with timestamps.
    """

    def __init__(
        self,
        rfid_host: str,
        rfid_port: Optional[int] = None,
        config_path: str = "config.json",
        **overrides,
    ) -> None:
        self._cfg = load_config(config_path, overrides)
        self._rfid_cfg = dict(self._cfg.get("rfid", {}) or {})
        if rfid_host:
            self._rfid_cfg["host"] = str(rfid_host).strip()
        if rfid_port is not None:
            self._rfid_cfg["port"] = int(rfid_port)

        timing_cfg = self._cfg.get("timing", {}) or {}
        self._tag_cooldown_seconds = float(timing_cfg.get("tag_cooldown_seconds") or 0.0)

        self._rfid_listener = RFIDListener(
            host=self._rfid_cfg.get("host", ""),
            port=int(self._rfid_cfg.get("port", 6000)),
            reconnect_seconds=float(self._rfid_cfg.get("reconnect_seconds") or 2.0),
            stale_seconds=float(self._rfid_cfg.get("stale_seconds") or 0.0),
            connect_timeout_seconds=float(self._rfid_cfg.get("connect_timeout_seconds") or 5.0),
            read_timeout_seconds=float(self._rfid_cfg.get("read_timeout_seconds") or 1.0),
            debounce_seconds=float(self._rfid_cfg.get("debounce_seconds") or 1.0),
            present_window_seconds=float(self._rfid_cfg.get("present_window_seconds") or 2.0),
            queue_maxsize=int(self._rfid_cfg.get("queue_maxsize") or 50),
        )

        self._running = False
        self._lock = threading.Lock()
        self._last_tag_event: Optional[TagEvent] = None
        self._last_error = ""
        self._tag_last_emitted: dict[str, float] = {}

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._rfid_listener.start()

    def stop(self) -> None:
        self._running = False
        self._rfid_listener.stop()

    def get_status(self) -> StationStatus:
        with self._lock:
            last_tag = self._last_tag_event
            last_error = self._last_error
        return StationStatus(
            running=self._running,
            rfid=self._rfid_listener.get_status(),
            primary_camera=None,
            secondary_camera=None,
            last_tag=last_tag,
            last_capture=None,
            last_error=last_error,
        )

    def read_next_tag(self, timeout_sec: Optional[float] = None) -> TagEvent:
        if not self._running:
            now = time.time()
            event = TagEvent(
                tag="",
                timestamp=now,
                timestamp_iso=_utc_iso(now),
                success=False,
                error="Station not running",
                rfid_connected=False,
                rfid_error="Station not running",
            )
            with self._lock:
                self._last_error = event.error
            return event

        deadline = None
        if timeout_sec is not None and timeout_sec > 0:
            deadline = time.time() + timeout_sec

        while self._running:
            if deadline is not None:
                remaining = deadline - time.time()
                if remaining <= 0:
                    now = time.time()
                    event = TagEvent(
                        tag="",
                        timestamp=now,
                        timestamp_iso=_utc_iso(now),
                        success=False,
                        error="Timed out waiting for tag",
                        rfid_connected=self._rfid_listener.get_status().connected,
                        rfid_error=self._rfid_listener.get_status().last_error,
                    )
                    with self._lock:
                        self._last_error = event.error
                    return event
                timeout = min(0.5, remaining)
            else:
                timeout = 0.5

            event = self._rfid_listener.read_next_event(timeout=timeout)
            if event is None:
                continue
            if not event.success:
                continue

            if self._tag_cooldown_seconds > 0:
                last_seen = self._tag_last_emitted.get(event.tag)
                if last_seen is not None and (time.time() - last_seen) < self._tag_cooldown_seconds:
                    continue

            with self._lock:
                self._last_tag_event = event
                self._last_error = ""
            self._tag_last_emitted[event.tag] = time.time()
            return event

        now = time.time()
        return TagEvent(
            tag="",
            timestamp=now,
            timestamp_iso=_utc_iso(now),
            success=False,
            error="Station stopped",
            rfid_connected=False,
            rfid_error="Station stopped",
        )
