import threading
import time

import cv2

from core.config import build_rtsp_url


class CameraReader:
    def __init__(self, name: str, cfg: dict, status: dict, status_lock, logger, reconnect_seconds: float, stale_seconds: float, fail_threshold: int):
        self.name = name
        self.cfg = cfg or {}
        self.status = status
        self.status_lock = status_lock
        self.logger = logger
        self.reconnect_seconds = reconnect_seconds
        self.stale_seconds = stale_seconds
        self.fail_threshold = fail_threshold

        self.latest_frame = None
        self.frame_lock = threading.Lock()

        self.capture = None
        self.running = False
        self.thread = None

        self.source = self._resolve_source()
        self.enabled = self.source is not None

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

    def get_frame(self):
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def _log_event(self, event_type: str, **fields) -> None:
        self.logger.log(event_type, **fields)

    def _run(self) -> None:
        was_connected = False
        was_reconnecting = False

        while self.running:
            if not self.enabled:
                with self.status_lock:
                    self.status["enabled"] = False
                    self.status["connected"] = False
                    self.status["reconnecting"] = False
                    self.status["last_error"] = f"{self.name} camera disabled (missing camera config)"
                if was_connected:
                    self._log_event(f"camera_{self.name}_lost", error="Camera disabled")
                    was_connected = False
                time.sleep(0.5)
                continue

            if self.capture is None or not self.capture.isOpened():
                with self.status_lock:
                    self.status["reconnecting"] = True
                    self.status["connected"] = False
                if not was_reconnecting:
                    self._log_event(f"camera_{self.name}_reconnecting", error=self.status.get("last_error") or "")
                    was_reconnecting = True

                ok_open = self._open_camera()
                now = time.time()
                with self.status_lock:
                    self.status["enabled"] = True
                    self.status["rtsp_url"] = str(self.source)
                    self.status["connected"] = bool(ok_open)
                    self.status["reconnecting"] = not ok_open
                    self.status["last_error"] = "" if ok_open else f"Failed to open stream: {self.source}"
                    if ok_open:
                        self.status["last_ok_ts"] = now
                        self.status["consecutive_failures"] = 0

                if not ok_open:
                    time.sleep(self.reconnect_seconds)
                    continue
                if not was_connected:
                    self._log_event(f"camera_{self.name}_connected", source=str(self.source))
                    was_connected = True
                    was_reconnecting = False

            try:
                ok, frame = self.capture.read()
            except Exception as exc:
                ok = False
                frame = None
                with self.status_lock:
                    self.status["connected"] = False
                    self.status["last_error"] = str(exc)

            if ok and frame is not None:
                with self.frame_lock:
                    self.latest_frame = frame
                now = time.time()
                with self.status_lock:
                    self.status["connected"] = True
                    self.status["reconnecting"] = False
                    self.status["last_error"] = ""
                    self.status["last_frame_time"] = now
                    self.status["last_frame_ts"] = now
                    self.status["last_ok_ts"] = now
                    self.status["consecutive_failures"] = 0
                if not was_connected:
                    self._log_event(f"camera_{self.name}_connected", source=str(self.source))
                    was_connected = True
                was_reconnecting = False
            else:
                with self.status_lock:
                    failures = int(self.status.get("consecutive_failures") or 0) + 1
                    self.status["consecutive_failures"] = failures
                    err = self.status.get("last_error") or "Failed to read frame"
                    self.status["last_error"] = err
                if failures >= self.fail_threshold:
                    try:
                        if self.capture is not None:
                            self.capture.release()
                    except Exception:
                        pass
                    self.capture = None
                    with self.status_lock:
                        self.status["connected"] = False
                        self.status["reconnecting"] = True
                    if was_connected:
                        self._log_event(f"camera_{self.name}_lost", error=err)
                        was_connected = False
                    if not was_reconnecting:
                        self._log_event(f"camera_{self.name}_reconnecting", error=err)
                        was_reconnecting = True
                    time.sleep(self.reconnect_seconds)
                    continue
                with self.status_lock:
                    self.status["connected"] = False
                    if not self.status.get("last_error"):
                        self.status["last_error"] = "Failed to read frame"
                time.sleep(0.1)

            if self.stale_seconds:
                with self.status_lock:
                    last_frame_ts = self.status.get("last_frame_ts")
                if last_frame_ts and (time.time() - last_frame_ts) > self.stale_seconds:
                    try:
                        if self.capture is not None:
                            self.capture.release()
                    except Exception:
                        pass
                    self.capture = None
                    with self.status_lock:
                        self.status["connected"] = False
                        self.status["reconnecting"] = True
                        self.status["last_error"] = f"{self.name} camera stale"
                    if was_connected:
                        self._log_event(f"camera_{self.name}_lost", error=f"{self.name} camera stale")
                        was_connected = False
                    if not was_reconnecting:
                        self._log_event(f"camera_{self.name}_reconnecting", error=f"{self.name} camera stale")
                        was_reconnecting = True
                    time.sleep(self.reconnect_seconds)


class CameraService:
    def __init__(self, camera_cfg: dict, secondary_cfg: dict, state, logger):
        camera_retry = float((camera_cfg or {}).get("retry_seconds") or 1.0)
        camera_stale = float((camera_cfg or {}).get("stale_seconds") or 0.0)
        camera_fail = int((camera_cfg or {}).get("fail_threshold") or 5)

        secondary_retry = float((secondary_cfg or {}).get("retry_seconds") or camera_retry)
        secondary_stale = float((secondary_cfg or {}).get("stale_seconds") or camera_stale)
        secondary_fail = int((secondary_cfg or {}).get("fail_threshold") or camera_fail)

        self.primary = CameraReader(
            "primary",
            camera_cfg,
            state.camera_status,
            state.lock,
            logger,
            camera_retry,
            camera_stale,
            camera_fail,
        )
        self.secondary = CameraReader(
            "secondary",
            secondary_cfg,
            state.secondary_camera_status,
            state.lock,
            logger,
            secondary_retry,
            secondary_stale,
            secondary_fail,
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
