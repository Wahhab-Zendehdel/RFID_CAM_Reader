import socket
import threading
import time

import listen_only

from core.config import get_env_float, get_env_int, get_env_str
from core.utils import normalize_tag


class RFIDDetector:
    def __init__(self, rfid_cfg: dict, state, logger, on_tag):
        self.state = state
        self.logger = logger
        self.on_tag = on_tag

        self.host = get_env_str("RFID_HOST") or str((rfid_cfg or {}).get("host") or "").strip()
        self.port = int(get_env_int("RFID_PORT") or (rfid_cfg or {}).get("port") or 6000)
        self.reconnect_seconds = float(get_env_float("RFID_RECONNECT_SECONDS") or (rfid_cfg or {}).get("reconnect_seconds") or 2.0)
        self.stale_seconds = float(get_env_float("RFID_STALE_SECONDS") or (rfid_cfg or {}).get("stale_seconds") or 0.0)
        self.connect_timeout_seconds = float(
            get_env_float("RFID_CONNECT_TIMEOUT_SECONDS")
            or (rfid_cfg or {}).get("connect_timeout_seconds")
            or 5.0
        )
        self.read_timeout_seconds = float(
            get_env_float("RFID_READ_TIMEOUT_SECONDS")
            or (rfid_cfg or {}).get("read_timeout_seconds")
            or 1.0
        )
        self.debounce_seconds = float(get_env_float("RFID_DEBOUNCE_SECONDS") or (rfid_cfg or {}).get("debounce_seconds") or 1.0)
        self.present_window_seconds = float(
            get_env_float("RFID_PRESENCE_WINDOW_SECONDS")
            or (rfid_cfg or {}).get("present_window_seconds")
            or 2.0
        )

        self.running = False
        self.thread = None

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

    def _update_status(self, **updates) -> None:
        with self.state.lock:
            self.state.rfid_status.update(updates)

    def _handle_tag(self, raw_tag: str, source: str) -> None:
        tag = normalize_tag(raw_tag)
        if not tag:
            return

        now = time.time()
        with self.state.lock:
            self.state.rfid_status["last_tag"] = tag
            self.state.rfid_status["last_tag_time"] = now
            self.state.rfid_status["last_tag_ts"] = now
            self.state.rfid_status["last_ok_ts"] = now
            self.state.rfid_status["last_tag_source"] = source

            last_seen = self.state.rfid_last_seen_by_tag.get(tag)
            if last_seen is not None and self.debounce_seconds > 0 and (now - last_seen) < self.debounce_seconds:
                return
            self.state.rfid_last_seen_by_tag[tag] = now

        self.logger.log("rfid", tag=tag, source=source)
        self.on_tag(tag, source)

    def _run(self) -> None:
        if not self.host:
            self._update_status(enabled=False)
            return

        buf = bytearray()
        was_connected = False
        was_reconnecting = False

        while self.running:
            sock = None
            try:
                self._update_status(
                    enabled=True,
                    host=self.host,
                    port=self.port,
                    last_error="",
                    reconnecting=True,
                )
                if not was_reconnecting:
                    self.logger.log("rfid_reconnecting", host=self.host, port=self.port)
                    was_reconnecting = True

                sock = listen_only.connect(self.host, self.port, timeout=self.connect_timeout_seconds)
                try:
                    sock.settimeout(self.read_timeout_seconds)
                except Exception:
                    pass

                now = time.time()
                self._update_status(
                    connected=True,
                    reconnecting=False,
                    last_error="",
                    last_ok_ts=now,
                    consecutive_failures=0,
                )
                if not was_connected:
                    self.logger.log("rfid_connected", host=self.host, port=self.port)
                    was_connected = True
                    was_reconnecting = False

                buf.clear()

                while self.running:
                    try:
                        chunk = sock.recv(4096)
                    except socket.timeout:
                        if self.stale_seconds:
                            with self.state.lock:
                                last_ok = self.state.rfid_status.get("last_ok_ts") or 0.0
                            if last_ok and (time.time() - last_ok) > self.stale_seconds:
                                raise TimeoutError("RFID stale (no data)")
                        continue

                    if not chunk:
                        raise ConnectionError("Device closed connection")

                    now = time.time()
                    self._update_status(last_ok_ts=now)
                    buf.extend(chunk)

                    for tag, source in listen_only.tags_from_buffer(buf):
                        self._handle_tag(tag, source)

            except Exception as exc:
                err = str(exc)
                with self.state.lock:
                    failures = int(self.state.rfid_status.get("consecutive_failures") or 0) + 1
                self._update_status(
                    connected=False,
                    reconnecting=True,
                    last_error=err,
                    consecutive_failures=failures,
                )
                self.logger.log("rfid_disconnected", error=err)
                self.logger.log("rfid_reconnecting", error=err)
                was_connected = False
                was_reconnecting = True
                time.sleep(self.reconnect_seconds)
            finally:
                try:
                    if sock is not None:
                        sock.close()
                except Exception:
                    pass
