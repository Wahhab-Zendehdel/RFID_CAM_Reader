from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Callable, Optional, List

from .common_camera import CameraService
from .common_config import deep_update, get_base_dir, load_config
from .common_label_ocr import LabelDetector, extract_target_digits
from .common_rfid import RFIDListener
from .models import CaptureResult, StationStatus, TagEvent


def _utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _apply_camera_input(cfg: dict, cam_input) -> dict:
    if cam_input is None:
        return cfg
    if isinstance(cam_input, dict):
        deep_update(cfg, cam_input)
        return cfg
    if isinstance(cam_input, int):
        cfg["index"] = cam_input
        return cfg
    cam_text = str(cam_input).strip()
    if not cam_text:
        return cfg
    if "://" in cam_text:
        cfg["rtsp_url"] = cam_text
    else:
        cfg["host"] = cam_text
    return cfg


class BascolStation:
    """
    Weighbridge station: listens to RFID and captures primary/secondary frames + OCR results.
    """

    def __init__(
        self,
        primary_cam,
        secondary_cam,
        rfid_host: str,
        rfid_port: Optional[int] = None,
        config_path: Optional[str] = None,
        validate_against_db: bool = False,
        **overrides,
    ) -> None:
        self._config_path = str(config_path or "")
        self._base_dir = get_base_dir(config_path)
        self._cfg = load_config(config_path, overrides)

        self._camera_cfg = dict(self._cfg.get("camera", {}) or {})
        self._secondary_cfg = dict(self._cfg.get("secondary_camera", {}) or {})
        _apply_camera_input(self._camera_cfg, primary_cam)
        _apply_camera_input(self._secondary_cfg, secondary_cam)

        self._rfid_cfg = dict(self._cfg.get("rfid", {}) or {})
        if rfid_host:
            self._rfid_cfg["host"] = str(rfid_host).strip()
        if rfid_port is not None:
            self._rfid_cfg["port"] = int(rfid_port)

        timing_cfg = self._cfg.get("timing", {}) or {}
        capture_cfg = self._cfg.get("capture", {}) or {}

        self._tag_cooldown_seconds = float(timing_cfg.get("tag_cooldown_seconds") or 0.0)
        self._retry_interval_seconds = float(capture_cfg.get("retry_interval_seconds") or 0.4)
        self._timeout_seconds = float(capture_cfg.get("timeout_seconds") or 30.0)
        self._target_digits = int(capture_cfg.get("target_digits") or 5)
        self._max_capture_attempts = int(capture_cfg.get("max_attempts") or 10)

        self._validate_against_db = bool(validate_against_db)

        self._camera_service = CameraService(self._camera_cfg, self._secondary_cfg)
        self._label_detector = LabelDetector(
            self._cfg.get("paper_detection", {}) or {},
            self._cfg.get("ocr", {}) or {},
            base_dir=self._base_dir,
        )
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
        self._last_capture_result: Optional[CaptureResult] = None
        self._last_error = ""
        self._callback: Optional[Callable[[CaptureResult], None]] = None
        self._tag_last_processed: dict[str, float] = {}

    def set_callback(self, fn: Optional[Callable[[CaptureResult], None]]) -> None:
        self._callback = fn

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._rfid_listener.start()
        self._camera_service.start()

    def stop(self) -> None:
        self._running = False
        self._rfid_listener.stop()
        self._camera_service.stop()

    def get_status(self) -> StationStatus:
        with self._lock:
            last_tag = self._last_tag_event
            last_capture = self._last_capture_result
            last_error = self._last_error
        return StationStatus(
            running=self._running,
            rfid=self._rfid_listener.get_status(),
            primary_camera=self._camera_service.get_primary_status(),
            secondary_camera=self._camera_service.get_secondary_status(),
            last_tag=last_tag,
            last_capture=last_capture,
            last_error=last_error,
        )

    def process_next(self, timeout_sec: Optional[float] = None) -> CaptureResult:
        if not self._running:
            now = time.time()
            result = CaptureResult(
                tag="",
                tag_source="",
                tag_timestamp=now,
                tag_timestamp_iso=_utc_iso(now),
                started_ts=now,
                finished_ts=now,
                success=False,
                message="Station not running",
                errors=["Station not running"],
            )
            self._update_last_capture(result)
            return result

        tag_event = self._wait_for_tag(timeout_sec)
        if tag_event is None:
            now = time.time()
            message = "Timed out waiting for tag" if self._running else "Station stopped"
            result = CaptureResult(
                tag="",
                tag_source="",
                tag_timestamp=now,
                tag_timestamp_iso=_utc_iso(now),
                started_ts=now,
                finished_ts=now,
                success=False,
                message=message,
                errors=[message],
            )
            self._update_last_capture(result)
            return result

        with self._lock:
            self._last_tag_event = tag_event

        result = self._process_capture(tag_event)
        self._update_last_capture(result)
        self._maybe_callback(result)
        return result

    def _update_last_capture(self, result: CaptureResult) -> None:
        with self._lock:
            self._last_capture_result = result
            self._last_error = result.errors[0] if result.errors else ""

    def _maybe_callback(self, result: CaptureResult) -> None:
        if not self._callback:
            return
        try:
            self._callback(result)
        except Exception:
            pass

    def _wait_for_tag(self, timeout_sec: Optional[float]) -> Optional[TagEvent]:
        deadline = None
        if timeout_sec is not None and timeout_sec > 0:
            deadline = time.time() + timeout_sec

        while self._running:
            if deadline is not None:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                timeout = min(0.5, remaining)
            else:
                timeout = 0.5

            event = self._rfid_listener.read_next_event(timeout=timeout)
            if event is None:
                continue

            with self._lock:
                self._last_tag_event = event

            if not event.success:
                continue

            if self._tag_cooldown_seconds > 0:
                last_seen = self._tag_last_processed.get(event.tag)
                if last_seen is not None and (time.time() - last_seen) < self._tag_cooldown_seconds:
                    continue

            return event

        return None

    def _process_capture(self, tag_event: TagEvent) -> CaptureResult:
        started = time.time()
        attempts = 0
        tags: List[str] = [tag_event.tag] if tag_event and tag_event.tag else []
        errors = []
        last_frame = None
        last_annotated = None
        last_warped = None
        last_text = ""
        message = ""
        number = ""

        while self._running:
            if (time.time() - started) > self._timeout_seconds:
                message = f"Timed out after {attempts} attempts"
                errors.append(message)
                break

            attempts += 1
            if self._max_capture_attempts and attempts > self._max_capture_attempts:
                message = f"Max attempts reached ({self._max_capture_attempts})"
                errors.append(message)
                break

            # collect any additional tags seen during capture (non-blocking)
            try:
                while True:
                    extra = self._rfid_listener.read_next_event(timeout=0)
                    if extra is None:
                        break
                    if extra.success and extra.tag and extra.tag not in tags:
                        tags.append(extra.tag)
            except Exception:
                pass

            frame = self._camera_service.get_primary_frame()
            last_frame = frame

            if frame is None:
                err = f"Primary frame not available (attempt {attempts})"
                errors.append(err)
                time.sleep(self._retry_interval_seconds)
                continue

            try:
                success, msg, text, annotated, warped = self._label_detector.process_frame(frame)
            except Exception as exc:
                message = f"Label detection failed: {exc}"
                errors.append(message)
                break

            message = msg
            last_text = text or ""
            last_annotated = annotated if annotated is not None else frame
            last_warped = warped

            if success:
                number = extract_target_digits(text or "", self._target_digits)
            else:
                number = ""

            if number:
                break

            time.sleep(self._retry_interval_seconds)

        secondary_frame = self._camera_service.get_secondary_frame()
        if secondary_frame is None:
            errors.append("Secondary frame not available")

        finished = time.time()
        if tag_event.tag:
            self._tag_last_processed[tag_event.tag] = finished

        # If we failed to get a number, append helpful device status errors
        if not number:
            try:
                pstat = self._camera_service.get_primary_status()
                if pstat and pstat.last_error:
                    errors.append(f"Primary camera error: {pstat.last_error}")
            except Exception:
                pass
            try:
                sstat = self._camera_service.get_secondary_status()
                if sstat and sstat.last_error:
                    errors.append(f"Secondary camera error: {sstat.last_error}")
            except Exception:
                pass
            try:
                rstat = self._rfid_listener.get_status()
                if rstat and rstat.last_error:
                    errors.append(f"RFID error: {rstat.last_error}")
            except Exception:
                pass

        return CaptureResult(
            tag=tag_event.tag,
            tag_source=tag_event.source,
            tag_timestamp=tag_event.timestamp,
            tag_timestamp_iso=tag_event.timestamp_iso,
            started_ts=started,
            finished_ts=finished,
            success=bool(number),
            number=number,
            raw_text=last_text,
            message=message,
            errors=errors,
            tags=tags,
            primary_image=last_frame,
            secondary_image=secondary_frame,
            label_image=last_warped,
            annotated_image=last_annotated,
            attempts=attempts,
        )
