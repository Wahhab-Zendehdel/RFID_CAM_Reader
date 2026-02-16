from __future__ import annotations

import queue
import re
import socket
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone
from typing import Callable, Iterator, Optional, Tuple

from .models import RFIDStatus, TagEvent


def utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def normalize_tag(tag: str) -> str:
    tag = (tag or "").strip().upper()
    return re.sub(r"[^0-9A-F]", "", tag)


def checksum_twos_complement(buf: bytes) -> int:
    s = sum(buf) & 0xFF
    return ((~s + 1) & 0xFF)


def verify_packet(frame: bytes) -> bool:
    if len(frame) < 5 or frame[0] != 0xA0:
        return False
    return checksum_twos_complement(frame[:-1]) == frame[-1]


def frames_from_stream(buf: bytearray) -> Iterator[bytes]:
    # total frame bytes = LEN + 2 (Head + Len)
    while True:
        idx = buf.find(b"\xA0")
        if idx == -1:
            buf.clear()
            return
        if idx > 0:
            del buf[:idx]
        if len(buf) < 2:
            return
        length = buf[1]
        total = length + 2
        if len(buf) < total:
            return
        frame = bytes(buf[:total])
        del buf[:total]
        yield frame


def short_records(buf: bytearray) -> Iterator[bytes]:
    """
    Handles the simple short records observed from this reader:
      - 14-byte: 0D 00 EE 00 20 00 <8-byte UID>
      - 8-byte:  07 00 EE 00 20 00 <2-byte UID>
    """
    while True:
        if len(buf) < 2:
            return
        head = buf[0]
        # Let A0-framed parsers handle this (do not desync by dropping 0xA0).
        if head == 0xA0:
            return
        if head == 0x0D and len(buf) >= 14:
            frame = bytes(buf[:14])
            del buf[:14]
            yield frame
            continue
        if head == 0x07 and len(buf) >= 8:
            frame = bytes(buf[:8])
            del buf[:8]
            yield frame
            continue
        # Head not recognized; let other parsers try
        return


def len_prefixed_records(buf: bytearray) -> Iterator[bytes]:
    """
    Some readers send frames where the first byte is total length, followed by
    0x00 0xEE 0x00 and data. Example observed: 11 00 EE 00 <EPC bytes> FF
    """
    while True:
        if len(buf) < 2:
            return
        head = buf[0]
        if head in (0x0D, 0x07, 0xA0):
            return
        total = head + 1
        if total <= 1:
            del buf[0]
            continue
        if len(buf) < total:
            return
        frame = bytes(buf[:total])
        del buf[:total]
        yield frame


def tag_from_len_prefixed(frame: bytes) -> Optional[Tuple[str, str]]:
    """
    Extract tag from length-prefixed frame that starts with:
    <len> 00 EE 00 <data> FF
    """
    if len(frame) < 6:
        return None
    if not (frame[1] == 0x00 and frame[2] == 0xEE and frame[3] == 0x00):
        return None
    uid = frame[4:-1] if len(frame) > 5 else b""
    if not uid:
        return None
    return uid.hex().upper(), "LEN"


def tag_from_short_record(frame: bytes) -> Optional[Tuple[str, str]]:
    """
    Convert a short record frame into (tag_hex, source).
    """
    if len(frame) == 14 and frame[0] == 0x0D:
        uid = frame[-8:]
        return uid.hex().upper(), "SHORT14"
    if len(frame) == 8 and frame[0] == 0x07:
        uid = frame[-2:]
        return uid.hex().upper(), "SHORT8"
    return None


def epc_from_a0_frame(frame: bytes) -> Optional[str]:
    """
    Extract EPC as hex from standard A0 responses (0x89 realtime, 0x90 buffer).
    Returns EPC hex or None.
    """
    if len(frame) < 5 or frame[0] != 0xA0:
        return None
    cmd = frame[3]

    if cmd == 0x89:
        length = frame[1]
        if length == 0x0A:
            return None
        epc_len = length - 7
        if epc_len <= 0:
            return None
        epc = frame[7 : 7 + epc_len]
        return epc.hex().upper() if epc else None

    if cmd == 0x90 and len(frame) >= 12:
        datalen = frame[6]
        if 7 + datalen + 4 > len(frame):
            return None
        data = frame[7 : 7 + datalen]
        if datalen < 4:
            return None
        epc = data[2:-2]
        return epc.hex().upper() if epc else None

    return None


def tags_from_buffer(buf: bytearray) -> Iterator[Tuple[str, str]]:
    """
    Parse the incoming stream buffer and yield (tag_hex, source) tuples.
    Uses:
      1) short_records (0x0D / 0x07)
      2) length-prefixed frames (len, 00 EE 00, data, FF)
      3) A0-framed packets (0xA0)
    """
    for frame in short_records(buf):
        parsed = tag_from_short_record(frame)
        if parsed:
            yield parsed

    for frame in len_prefixed_records(buf):
        parsed = tag_from_len_prefixed(frame)
        if parsed:
            yield parsed

    for frame in frames_from_stream(buf):
        if not verify_packet(frame):
            continue
        epc = epc_from_a0_frame(frame)
        if epc:
            yield epc, f"A0_{frame[3]:02X}"


def connect(host: str, port: int, timeout: float = 5.0) -> socket.socket:
    sock = socket.create_connection((host, port), timeout=timeout)
    sock.settimeout(1.0)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    except OSError:
        pass
    return sock


class RFIDListener:
    def __init__(
        self,
        host: str,
        port: int,
        reconnect_seconds: float = 2.0,
        stale_seconds: float = 0.0,
        connect_timeout_seconds: float = 5.0,
        read_timeout_seconds: float = 1.0,
        debounce_seconds: float = 1.0,
        present_window_seconds: float = 2.0,
        queue_maxsize: int = 50,
        logger: Optional[Callable[..., None]] = None,
    ) -> None:
        self.host = (host or "").strip()
        self.port = int(port or 6000)
        self.reconnect_seconds = float(reconnect_seconds or 0.0)
        self.stale_seconds = float(stale_seconds or 0.0)
        self.connect_timeout_seconds = float(connect_timeout_seconds or 5.0)
        self.read_timeout_seconds = float(read_timeout_seconds or 1.0)
        self.debounce_seconds = float(debounce_seconds or 0.0)
        self.present_window_seconds = float(present_window_seconds or 0.0)
        self.logger = logger

        self._status_lock = threading.Lock()
        self._status = RFIDStatus(
            enabled=bool(self.host),
            host=self.host,
            port=self.port,
            connected=False,
            reconnecting=False,
            last_ok_ts=None,
            last_tag_ts=None,
            last_tag="",
            last_tag_source="",
            last_error="",
            consecutive_failures=0,
        )

        self._queue: queue.Queue[TagEvent] = queue.Queue(maxsize=queue_maxsize)
        self._last_seen_by_tag: dict[str, float] = {}
        self._present_until_by_tag: dict[str, float] = {}

        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def read_next_event(self, timeout: Optional[float] = None) -> Optional[TagEvent]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_status(self) -> RFIDStatus:
        with self._status_lock:
            return replace(self._status)

    def _log(self, event_type: str, **fields) -> None:
        if not self.logger:
            return
        try:
            self.logger(event_type, **fields)
        except Exception:
            pass

    def _update_status(self, **updates) -> None:
        with self._status_lock:
            for key, value in updates.items():
                setattr(self._status, key, value)

    def _handle_tag(self, raw_tag: str, source: str) -> None:
        tag = normalize_tag(raw_tag)
        if not tag:
            return

        now = time.time()
        last_seen = self._last_seen_by_tag.get(tag)
        if last_seen is not None and self.debounce_seconds > 0 and (now - last_seen) < self.debounce_seconds:
            return

        present_until = self._present_until_by_tag.get(tag)
        if present_until is not None and self.present_window_seconds > 0 and now < present_until:
            return

        self._last_seen_by_tag[tag] = now
        if self.present_window_seconds > 0:
            self._present_until_by_tag[tag] = now + self.present_window_seconds

        self._update_status(
            last_tag=tag,
            last_tag_ts=now,
            last_tag_source=source,
            last_ok_ts=now,
        )

        event = TagEvent(
            tag=tag,
            timestamp=now,
            timestamp_iso=utc_iso(now),
            source=source,
            raw_tag=raw_tag,
            success=True,
            error="",
            rfid_connected=True,
            rfid_error=self.get_status().last_error,
        )

        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self._update_status(last_error="RFID queue full")

        self._log("rfid_tag", tag=tag, source=source)

    def _run(self) -> None:
        if not self.host:
            self._update_status(enabled=False, connected=False, reconnecting=False, last_error="RFID disabled")
            return

        buf = bytearray()

        while self._running:
            sock: Optional[socket.socket] = None
            try:
                self._update_status(
                    enabled=True,
                    host=self.host,
                    port=self.port,
                    last_error="",
                    reconnecting=True,
                )

                sock = connect(self.host, self.port, timeout=self.connect_timeout_seconds)
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
                self._log("rfid_connected", host=self.host, port=self.port)

                buf.clear()

                while self._running:
                    try:
                        chunk = sock.recv(4096)
                    except socket.timeout:
                        if self.stale_seconds:
                            last_ok = self.get_status().last_ok_ts or 0.0
                            if last_ok and (time.time() - last_ok) > self.stale_seconds:
                                raise TimeoutError("RFID stale (no data)")
                        continue

                    if not chunk:
                        raise ConnectionError("Device closed connection")

                    now = time.time()
                    self._update_status(last_ok_ts=now)
                    buf.extend(chunk)

                    for tag, source in tags_from_buffer(buf):
                        self._handle_tag(tag, source)

            except Exception as exc:
                err = str(exc)
                failures = self.get_status().consecutive_failures + 1
                self._update_status(
                    connected=False,
                    reconnecting=True,
                    last_error=err,
                    consecutive_failures=failures,
                )
                self._log("rfid_disconnected", error=err)
                time.sleep(self.reconnect_seconds)
            finally:
                try:
                    if sock is not None:
                        sock.close()
                except Exception:
                    pass
