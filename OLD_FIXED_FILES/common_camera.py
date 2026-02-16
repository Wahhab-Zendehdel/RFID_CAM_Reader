from __future__ import annotations

import threading
import time
from dataclasses import replace
from typing import Optional

import cv2

from .common_config import build_rtsp_url
from .models import CameraStatus


class CameraReader:
    def __init__(
        self,
        name: str,
        cfg: dict,
        status: CameraStatus,
        status_lock: threading.Lock,
        reconnect_seconds: float,
        stale_seconds: float,
        fail_threshold: int,
        logger=None,
    ) -> None:
        self.name = name
        self.cfg = cfg or {}
        self.status = status
        self.status_lock = status_lock
        self.reconnect_seconds = reconnect_seconds
        self.stale_seconds = stale_seconds
        self.fail_threshold = fail_threshold
        self.logger = logger

        self.latest_frame = None
        self.frame_lock = threading.Lock()

        self.capture = None
        self.running = False
        self.thread: Optional[threading.Thread] = None

        self.source = self._resolve_source()
        self.enabled = self.source is not None

        with self.status_lock:
            self.status.name = self.name
            self.status.enabled = self.enabled
            self.status.source = str(self.source) if self.source is not None else ""

    def _resolve_source(self):
        rtsp_url = str(self.cfg.get("rtsp_url") or "").strip()
        if rtsp_url:
            return rtsp_url

        index = self.cfg.get("index")
        if index is not None:
            return int(index)

        derived = build_rtsp_url(self.cfg)
        if derived:
            return derived

        return None

    def _apply_camera_settings(self, cap) -> None:
        width = self.cfg.get("width")
        height = self.cfg.get("height")
        fps = self.cfg.get("fps")
        auto_exposure = self.cfg.get("auto_exposure")
        exposure = self.cfg.get("exposure")

        try:
            if width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
            if height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
            if fps:
                cap.set(cv2.CAP_PROP_FPS, float(fps))
            if auto_exposure is not None:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, float(auto_exposure))
            if exposure is not None:
                cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
        except Exception:
            pass

    def _open_camera(self) -> bool:
        if not self.enabled:
            return False
        try:
            if self.capture is not None:
                self.capture.release()
        except Exception:
            pass

        self.capture = cv2.VideoCapture(self.source)
        try:
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self._apply_camera_settings(self.capture)
        return bool(self.capture.isOpened())

    def start(self) -> None:
        if self.thread is not None:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
        try:
            if self.capture is not None:
                self.capture.release()
        except Exception:
            pass

    def get_frame(self):
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def get_status(self) -> CameraStatus:
        with self.status_lock:
            return replace(self.status)

    def _log(self, event_type: str, **fields) -> None:
        if not self.logger:
            return
        try:
            self.logger(event_type, **fields)
        except Exception:
            pass

    def _update_status(self, **updates) -> None:
        with self.status_lock:
            for key, value in updates.items():
                setattr(self.status, key, value)

    def _run(self) -> None:
        was_connected = False
        was_reconnecting = False

        while self.running:
            if not self.enabled:
                self._update_status(
                    enabled=False,
                    connected=False,
                    reconnecting=False,
                    last_error=f"{self.name} camera disabled",
                )
                if was_connected:
                    self._log(f"camera_{self.name}_lost", error="Camera disabled")
                    was_connected = False
                time.sleep(0.5)
                continue

            if self.capture is None or not self.capture.isOpened():
                self._update_status(reconnecting=True, connected=False)
                if not was_reconnecting:
                    self._log(f"camera_{self.name}_reconnecting", error=self.status.last_error or "")
                    was_reconnecting = True

                ok_open = self._open_camera()
                now = time.time()
                self._update_status(
                    enabled=True,
                    source=str(self.source),
                    connected=bool(ok_open),
                    reconnecting=not ok_open,
                    last_error="" if ok_open else f"Failed to open stream: {self.source}",
                )
                if ok_open:
                    self._update_status(last_ok_ts=now, consecutive_failures=0)

                if not ok_open:
                    time.sleep(self.reconnect_seconds)
                    continue
                if not was_connected:
                    self._log(f"camera_{self.name}_connected", source=str(self.source))
                    was_connected = True
                    was_reconnecting = False

            try:
                ok, frame = self.capture.read()
            except Exception as exc:
                ok = False
                frame = None
                self._update_status(connected=False, last_error=str(exc))

            if ok and frame is not None:
                with self.frame_lock:
                    self.latest_frame = frame
                now = time.time()
                self._update_status(
                    connected=True,
                    reconnecting=False,
                    last_error="",
                    last_frame_ts=now,
                    last_ok_ts=now,
                    consecutive_failures=0,
                )
                if not was_connected:
                    self._log(f"camera_{self.name}_connected", source=str(self.source))
                    was_connected = True
                was_reconnecting = False
            else:
                failures = self.status.consecutive_failures + 1
                self._update_status(consecutive_failures=failures)
                err = self.status.last_error or "Failed to read frame"
                self._update_status(last_error=err)
                if failures >= self.fail_threshold:
                    try:
                        if self.capture is not None:
                            self.capture.release()
                    except Exception:
                        pass
                    self.capture = None
                    self._update_status(connected=False, reconnecting=True)
                    if was_connected:
                        self._log(f"camera_{self.name}_lost", error=err)
                        was_connected = False
                    if not was_reconnecting:
                        self._log(f"camera_{self.name}_reconnecting", error=err)
                        was_reconnecting = True
                    time.sleep(self.reconnect_seconds)
                    continue
                self._update_status(connected=False)
                time.sleep(0.1)

            if self.stale_seconds:
                last_frame_ts = self.status.last_frame_ts
                if last_frame_ts and (time.time() - last_frame_ts) > self.stale_seconds:
                    try:
                        if self.capture is not None:
                            self.capture.release()
                    except Exception:
                        pass
                    self.capture = None
                    self._update_status(
                        connected=False,
                        reconnecting=True,
                        last_error=f"{self.name} camera stale",
                    )
                    if was_connected:
                        self._log(f"camera_{self.name}_lost", error=f"{self.name} camera stale")
                        was_connected = False
                    if not was_reconnecting:
                        self._log(f"camera_{self.name}_reconnecting", error=f"{self.name} camera stale")
                        was_reconnecting = True
                    time.sleep(self.reconnect_seconds)


class CameraService:
    def __init__(self, camera_cfg: dict, secondary_cfg: dict, logger=None) -> None:
        camera_retry = float((camera_cfg or {}).get("retry_seconds") or 1.0)
        camera_stale = float((camera_cfg or {}).get("stale_seconds") or 0.0)
        camera_fail = int((camera_cfg or {}).get("fail_threshold") or 5)

        secondary_retry = float((secondary_cfg or {}).get("retry_seconds") or camera_retry)
        secondary_stale = float((secondary_cfg or {}).get("stale_seconds") or camera_stale)
        secondary_fail = int((secondary_cfg or {}).get("fail_threshold") or camera_fail)

        self._status_lock = threading.Lock()
        self._primary_status = CameraStatus(name="primary")
        self._secondary_status = CameraStatus(name="secondary")

        self.primary = CameraReader(
            "primary",
            camera_cfg,
            self._primary_status,
            self._status_lock,
            reconnect_seconds=camera_retry,
            stale_seconds=camera_stale,
            fail_threshold=camera_fail,
            logger=logger,
        )
        self.secondary = CameraReader(
            "secondary",
            secondary_cfg,
            self._secondary_status,
            self._status_lock,
            reconnect_seconds=secondary_retry,
            stale_seconds=secondary_stale,
            fail_threshold=secondary_fail,
            logger=logger,
        )

    def start(self) -> None:
        self.primary.start()
        self.secondary.start()

    def stop(self) -> None:
        self.primary.stop()
        self.secondary.stop()

    def get_primary_frame(self):
        return self.primary.get_frame()

    def get_secondary_frame(self):
        return self.secondary.get_frame()

    def get_primary_status(self) -> CameraStatus:
        return self.primary.get_status()

    def get_secondary_status(self) -> CameraStatus:
        return self.secondary.get_status()
