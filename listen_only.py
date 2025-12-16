"""
Auto-start RFID listener: connects to the reader over LAN (hardcoded host/port),
prints tag IDs, and exposes a hook where you can trigger other Python code.
No CLI args required; just run:  python auto_rfid_listener.py
"""

import socket
import time
from typing import Iterator, Optional, Tuple

# --- CONFIGURE YOUR READER HERE ---
HOST = "192.168.1.2"
PORT = 6000
# ----------------------------------


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
        # Let A0-framed parsers handle this (don't desync by dropping 0xA0).
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
    Some readers send frames where the first byte is total length, followed by 0x00 0xEE 0x00 and data.
    Example observed: 11 00 EE 00 <EPC bytes> FF
    """
    while True:
        if len(buf) < 2:
            return
        head = buf[0]
        if head in (0x0D, 0x07, 0xA0):  # let other parsers handle those
            return
        total = head + 1  # include the length byte itself
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
    Extract tag from length-prefixed frame that starts with: <len> 00 EE 00 <data> FF
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
        epc = frame[7:7 + epc_len]
        return epc.hex().upper() if epc else None

    if cmd == 0x90 and len(frame) >= 12:
        datalen = frame[6]
        if 7 + datalen + 4 > len(frame):
            return None
        data = frame[7:7 + datalen]  # PC(2) + EPC + CRC(2)
        if datalen < 4:
            return None
        epc = data[2:-2]
        return epc.hex().upper() if epc else None

    return None


def tags_from_buffer(buf: bytearray) -> Iterator[Tuple[str, str]]:
    """
    Parse the incoming stream buffer and yield (tag_hex, source) tuples.
    Uses the same approach as `main()`:
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


def decode_realtime_epc(frame: bytes) -> Optional[str]:
    if frame[3] != 0x89:
        return None
    length = frame[1]
    if length == 0x0A:  # end-of-round status packet
        return None
    epc_len = length - 7
    if epc_len <= 0:
        return None
    freqant = frame[4]
    pc = frame[5:7]
    epc = frame[7:7 + epc_len]
    rssi = frame[7 + epc_len]
    ant_id = freqant & 0x03
    return f"[RT] EPC={epc.hex().upper()} PC={pc.hex().upper()} RSSI(raw)={rssi} ANT={ant_id}"


def decode_buffer_epc(frame: bytes) -> Optional[str]:
    if frame[3] != 0x90 or len(frame) < 12:
        return None
    tagcount = int.from_bytes(frame[4:6], "big", signed=False)
    datalen = frame[6]
    if 7 + datalen + 4 > len(frame):
        return None
    data = frame[7:7 + datalen]
    if datalen < 4:
        return None
    pc = data[:2]
    epc = data[2:-2]
    rssi_idx = 7 + datalen
    rssi = frame[rssi_idx]
    freqant = frame[-3]
    invcnt = frame[-2]
    ant_id = freqant & 0x03
    return f"[BUF] tags={tagcount} EPC={epc.hex().upper()} PC={pc.hex().upper()} RSSI(raw)={rssi} ANT={ant_id} COUNT={invcnt}"


def on_tag(uid_hex: str):
    """
    Hook for your project. This is called whenever a tag UID is seen in the
    14-byte records. Replace the print with your own logic.
    """
    print(f"TAG DETECTED: {uid_hex}")
    # Example: trigger another function/process here.
    # my_other_function(uid_hex)


def connect(host: str, port: int, timeout: float = 5.0) -> socket.socket:
    s = socket.create_connection((host, port), timeout=timeout)
    s.settimeout(1.0)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    except OSError:
        pass
    return s


def main():
    while True:
        try:
            print(f"Connecting to {HOST}:{PORT} ...")
            s = connect(HOST, PORT)
            print("Connected. Auto-listening.")

            buf = bytearray()
            while True:
                try:
                    chunk = s.recv(4096)
                except socket.timeout:
                    continue

                if not chunk:
                    print("Device closed connection.")
                    break

                # Debug visibility of inbound data
                print(f"RX {len(chunk)} bytes: {chunk.hex().upper()}")

                buf.extend(chunk)

                # Handle simple short records (14-byte or 8-byte)
                for frame in short_records(buf):
                    if len(frame) == 14:
                        uid = frame[-8:]
                    elif len(frame) == 8:
                        uid = frame[-2:]
                    else:
                        uid = frame[6:]
                    uid_hex = uid.hex().upper()
                    print(f"[SHORT] FRAME={frame.hex().upper()} UID({len(uid)}B)={uid_hex}")
                    on_tag(uid_hex)

                # Handle standard A0-framed responses if they appear
                for frame in frames_from_stream(buf):
                    if not verify_packet(frame):
                        print("BADCHK FRAME:", frame.hex().upper())
                        continue
                    rt = decode_realtime_epc(frame)
                    if rt:
                        print(rt)
                        continue
                    bf = decode_buffer_epc(frame)
                    if bf:
                        print(bf)
                        continue

        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
        except (OSError, ConnectionError) as e:
            print(f"Socket error: {e!r}")
        finally:
            try:
                s.close()
            except Exception:
                pass

        # Auto-reconnect after a short pause
        time.sleep(1.0)


if __name__ == "__main__":
    main()
