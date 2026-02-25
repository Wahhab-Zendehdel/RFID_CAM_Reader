from __future__ import annotations

import sys
import os

# Add parent directory to path so we can import lib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.bascol_station import BascolStation
from lib.sangshekan_station import SangShekanStation
from lib.common_config import load_config, load_env_file, load_stations
from lib.common_db import init_db, store_result, update_result
import multiprocessing
from typing import Any, Dict, Optional
import sqlite3
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
import threading

import json
import cv2
import time
from datetime import datetime


load_env_file()


def _load_db_cfg() -> Dict[str, Any]:
    cfg = load_config()
    return cfg.get("db") or {}


def _save_image_to_file(image, image_type: str, tag: str) -> str:
    """
    Save image to disk and return the URL.
    image_type: 'primary', 'secondary', or 'label'
    tag: the RFID tag name
    Returns: URL like http://127.0.0.1:3000/images/TAG001_primary_1769581721.123.jpg
    """
    if image is None:
        return ""
    try:
        os.makedirs('images', exist_ok=True)
        timestamp = int(time.time() * 1000)  # milliseconds
        filename = f"{tag}_{image_type}_{timestamp}.jpg"
        filepath = os.path.join('images', filename)
        
        cv2.imwrite(filepath, image)
        
        # Return URL that points to image server
        image_server_url = os.environ.get("IMAGE_SERVER_URL", "http://127.0.0.1:3000")
        url = f"{image_server_url}/images/{filename}"
        return url
    except Exception as e:
        print(f"  âœ— Failed to save {image_type} image: {e}")
        return ""


def _coerce_int(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return int(s)
        except Exception:
            return None
    try:
        return int(value)
    except Exception:
        return None


def _coerce_float(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def _resolve_station_session_timeout_seconds(conf: Dict[str, Any], runtime_cfg: Optional[Dict[str, Any]] = None) -> float:
    station_timeout = _coerce_float(conf.get("TAG_SESSION_TIMEOUT_SECONDS"))
    if station_timeout is None:
        station_timeout = _coerce_float(conf.get("tag_session_timeout_seconds"))
    if station_timeout is not None and station_timeout > 0:
        return station_timeout
    return _resolve_tag_session_timeout_seconds(runtime_cfg)


def _pick_station_camera(
    cameras: list[Dict[str, Any]],
    device_index: int,
) -> Optional[Dict[str, Any]]:
    if not cameras:
        return None
    return cameras[device_index % len(cameras)]


def _build_bascol_device_configs(conf: Dict[str, Any]) -> list[Dict[str, Any]]:
    primary_cameras = [dict(item) for item in (conf.get("primary_cameras") or []) if isinstance(item, dict)]
    secondary_cameras = [dict(item) for item in (conf.get("secondary_cameras") or []) if isinstance(item, dict)]
    rfid_devices = [dict(item) for item in (conf.get("rfid_devices") or []) if isinstance(item, dict)]

    devices: list[Dict[str, Any]] = []
    for idx, rfid in enumerate(rfid_devices):
        rfid_host = str(rfid.get("host") or "").strip()
        if not rfid_host:
            continue

        selected_primary = _pick_station_camera(primary_cameras, idx)
        selected_secondary = _pick_station_camera(secondary_cameras, idx)

        devices.append(
            {
                "index": idx,
                "rfid": rfid,
                "primary_camera": selected_primary,
                "secondary_camera": selected_secondary,
                "rfid_device_id": _coerce_int(rfid.get("id")),
                "primary_cam_id": _coerce_int((selected_primary or {}).get("id")),
                "secondary_cam_id": _coerce_int((selected_secondary or {}).get("id")),
            }
        )
    return devices


def _build_sangshekan_rfid_configs(conf: Dict[str, Any]) -> list[Dict[str, Any]]:
    rfid_devices = [dict(item) for item in (conf.get("rfid_devices") or []) if isinstance(item, dict)]
    devices: list[Dict[str, Any]] = []
    for idx, rfid in enumerate(rfid_devices):
        host = str(rfid.get("host") or "").strip()
        if not host:
            continue
        devices.append(
            {
                "index": idx,
                "rfid_device_id": _coerce_int(rfid.get("id")),
                "host": host,
                "port": _coerce_int(rfid.get("port")),
            }
        )
    return devices




def _init_db(db_path: str = "data/results.db") -> None:
    """Create SQLite DB and results table if not exists. Ensures parent dir."""
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        # Use WAL for better concurrency
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
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
        conn.commit()
    finally:
        conn.close()


def _store_result(payload: Dict[str, Any], db_path: str = "data/results.db") -> None:
    """Insert a payload into the results table. Serializes tags/errors as JSON."""
    try:
        conn = sqlite3.connect(db_path, timeout=30)
        cur = conn.cursor()
        tags_json = json.dumps(payload.get("tags") or [])
        errors_val = payload.get("errors")
        errors_json = json.dumps(errors_val) if errors_val is not None else None

        cur.execute(
            """
            INSERT INTO results (
                datetime, station, station_id, station_type, tags, number,
                primary_image_url, secondary_image_url, label_image_url,
                rfid_device_id, primary_cam_id, secondary_cam_id,
                errors
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            ),
        )
        conn.commit()
    except Exception as e:
        print(f"âœ— Failed to store result in DB: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass

def run_bascol_demo():
    """
    Continuous demo: listen for RFID tags, process each with camera capture,
    and store results in the local database regardless of success/failure.
    Runs indefinitely until stopped.
    """
    # Load DB config and initialize (will verify MySQL connectivity or create sqlite table)
    db_cfg = _load_db_cfg()
    runtime_cfg = load_config()
    session_timeout_seconds = _resolve_tag_session_timeout_seconds(runtime_cfg)
    tag_sessions: Dict[str, Dict[str, Any]] = {}
    try:
        init_db(db_cfg)
    except Exception as e:
        print(f"âœ— DB initialization warning: {e}")

    station = BascolStation(
        primary_cam="192.168.1.3",
        secondary_cam="192.168.1.201",
        rfid_host="192.168.1.2",
        rfid_port=6000,
        timing={"tag_cooldown_seconds": 0.0},
    )
    station.start()
    print("ًں“چ Bascol Station started. Listening for RFID tags...")
    print("   Running indefinitely. Press Ctrl+C to stop.\n")
    
    try:
        while True:
            try:
                # Process captures continuously
                # timeout_sec=None means wait indefinitely for tags
                print("âڈ³ Waiting for RFID tag...")
                result = station.process_next(timeout_sec=None)
                
                print("\n" + "=" * 70)
                print(f"ًں“¦ Capture Result (attempt {result.attempts}):")
                print(f"   Tags: {result.tags}")
                print(f"   Success: {result.success}")
                print(f"   Number: {result.number}")
                print(f"   Message: {result.message}")
                if result.errors:
                    print(f"   Errors: {result.errors}")
                print("=" * 70)
                
                _persist_bascol_result_with_tag_session(
                    result=result,
                    station_name="bascol",
                    db_cfg=db_cfg,
                    tag_sessions=tag_sessions,
                    session_timeout_seconds=session_timeout_seconds,
                    station_type="bascol",
                )
                print("âœ“ Results stored.\n")
                
            except Exception as e:
                print(f"\nâœ— Error processing capture: {e}")
                import traceback
                traceback.print_exc()
                continue
                
    except KeyboardInterrupt:
        print("\n\nًں›‘ Stopping Bascol Station...")
    except Exception as e:
        print(f"\nâœ— Fatal error in demo: {e}")
        import traceback
        traceback.print_exc()
    finally:
        station.stop()
        print("âœ“ Station stopped.")


def run_sangshekan_demo():
    station = SangShekanStation(
        rfid_host="192.168.1.2",
        rfid_port=6000,
    )
    station.start()
    try:
        event = station.read_next_tag()
        print("SangShekan tag:")
        print(event)
    finally:
        station.stop()
    


def _save_and_build_bascol_payload(
    result,
    station_name: str,
    station_id: Optional[int] = None,
    station_type: Optional[str] = None,
    rfid_device_id: Optional[int] = None,
    primary_cam_id: Optional[int] = None,
    secondary_cam_id: Optional[int] = None,
) -> Dict[str, Any]:
    primary_url = _save_image_to_file(result.primary_image, "primary", result.tag or station_name)
    secondary_url = _save_image_to_file(result.secondary_image, "secondary", result.tag or station_name)
    label_url = _save_image_to_file(result.label_image, "label", result.tag or station_name)

    payload = {
        "datetime": datetime.utcnow().isoformat(),
        "station": station_name,
        "station_id": station_id,
        "station_type": station_type,
        "tags": result.tags or ([result.tag] if result.tag else []),
        "number": result.number,
        "primary_image_url": primary_url or None,
        "secondary_image_url": secondary_url or None,
        "label_image_url": label_url or None,
        "rfid_device_id": rfid_device_id,
        "primary_cam_id": primary_cam_id,
        "secondary_cam_id": secondary_cam_id,
        "message": result.message,
        "errors": result.errors or None,
    }
    return payload


def _resolve_tag_session_timeout_seconds(cfg: Optional[Dict[str, Any]] = None) -> float:
    timing_cfg = ((cfg or {}).get("timing") or {}) if cfg else {}
    raw_value = timing_cfg.get("tag_session_timeout_seconds")
    try:
        timeout_seconds = float(raw_value if raw_value is not None else 10.0)
    except Exception:
        timeout_seconds = 10.0
    return timeout_seconds if timeout_seconds > 0 else 10.0


def _cleanup_expired_tag_sessions(tag_sessions: Dict[str, Dict[str, Any]], now_ts: float) -> None:
    stale_tags = [
        tag
        for tag, state in tag_sessions.items()
        if float(state.get("expires_at") or 0.0) <= now_ts
    ]
    for tag in stale_tags:
        tag_sessions.pop(tag, None)


def _persist_bascol_result_with_tag_session(
    result,
    station_name: str,
    db_cfg: Dict[str, Any],
    tag_sessions: Dict[str, Dict[str, Any]],
    session_timeout_seconds: float,
    logger: Optional[logging.Logger] = None,
    station_id: Optional[int] = None,
    station_type: Optional[str] = None,
    rfid_device_id: Optional[int] = None,
    primary_cam_id: Optional[int] = None,
    secondary_cam_id: Optional[int] = None,
) -> None:
    payload = _save_and_build_bascol_payload(
        result,
        station_name,
        station_id=station_id,
        station_type=station_type,
        rfid_device_id=rfid_device_id,
        primary_cam_id=primary_cam_id,
        secondary_cam_id=secondary_cam_id,
    )

    tag = str(result.tag or "").strip()
    if not tag:
        store_result(payload, db_cfg)
        return

    now_ts = float(getattr(result, "finished_ts", None) or time.time())
    _cleanup_expired_tag_sessions(tag_sessions, now_ts)
    payload_complete = bool(str(payload.get("number") or "").strip())
    state = tag_sessions.get(tag)
    in_active_window = state is not None and now_ts < float(state.get("expires_at") or 0.0)

    if not in_active_window:
        record_id = store_result(payload, db_cfg)
        tag_sessions[tag] = {
            "record_id": record_id,
            "expires_at": now_ts + session_timeout_seconds,
            "complete": payload_complete,
        }
        if logger is not None:
            logger.info(
                "session created tag=%s record_id=%s complete=%s",
                tag,
                record_id,
                payload_complete,
            )
        return

    state["expires_at"] = now_ts + session_timeout_seconds
    record_id = state.get("record_id")
    has_complete_data = bool(state.get("complete"))

    # Keep a complete row from being overwritten by a later incomplete read.
    if has_complete_data and not payload_complete:
        if logger is not None:
            logger.info("session refreshed tag=%s record_id=%s complete_data_kept=true", tag, record_id)
        return

    if record_id is None:
        recovered_id = store_result(payload, db_cfg)
        state["record_id"] = recovered_id
        state["complete"] = payload_complete
        if logger is not None:
            logger.warning("session missing record_id tag=%s recovered_record_id=%s", tag, recovered_id)
        return

    updated = update_result(int(record_id), payload, db_cfg)
    if not updated:
        if logger is not None:
            logger.warning("session update failed tag=%s record_id=%s", tag, record_id)
        return

    state["complete"] = has_complete_data or payload_complete
    if logger is not None:
        logger.info(
            "session updated tag=%s record_id=%s complete=%s",
            tag,
            record_id,
            bool(state["complete"]),
        )


def _setup_logger(station_name: str, logs_dir: str = "logs") -> logging.Logger:
    """Create and return a logger for a station. Writes to logs/{station_name}.log with rotation.

    The logger also logs to stdout for convenience.
    """
    os.makedirs(logs_dir, exist_ok=True)
    logger_name = f"station.{station_name}"
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # File handler with rotation
    logfile = os.path.join(logs_dir, f"{station_name}.log")
    fh = RotatingFileHandler(logfile, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] [pid:%(process)d] %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Avoid propagating to root handlers
    logger.propagate = False
    return logger


def _build_sangshekan_payload(
    event,
    station_name: str,
    station_id: Optional[int] = None,
    station_type: Optional[str] = None,
    rfid_device_id: Optional[int] = None,
    primary_cam_id: Optional[int] = None,
    secondary_cam_id: Optional[int] = None,
) -> Dict[str, Any]:
    # Only include the clean fields; other fields should remain null for sangshekan
    return {
        "datetime": datetime.utcnow().isoformat(),
        "station": station_name,
        "station_id": station_id,
        "station_type": station_type,
        "tags": [event.tag] if getattr(event, "tag", None) else [],
        "number": None,
        "primary_image_url": None,
        "secondary_image_url": None,
        "label_image_url": None,
        "rfid_device_id": rfid_device_id,
        "primary_cam_id": primary_cam_id,
        "secondary_cam_id": secondary_cam_id,
        "message": getattr(event, "message", getattr(event, "error", "")),
        "errors": getattr(event, "error", None),
    }


def _bascol_worker(conf: Dict[str, Any]) -> None:
    name = conf.get("name") or "bascol"
    station_id = _coerce_int(conf.get("id"))
    station_type = conf.get("type") or "bascol"
    logger = _setup_logger(name)
    runtime_cfg = load_config()
    session_timeout_seconds = _resolve_station_session_timeout_seconds(conf, runtime_cfg)
    tag_sessions: Dict[str, Dict[str, Any]] = {}
    session_lock = threading.Lock()
    stop_event = threading.Event()
    logger.info("starting bascol worker session_timeout_seconds=%s", session_timeout_seconds)

    device_confs = _build_bascol_device_configs(conf)
    if not device_confs:
        logger.error("no RFID devices configured for bascol station")
        return

    db_cfg: Dict[str, Any] = {}
    try:
        db_cfg = _load_db_cfg()
        init_db(db_cfg)
    except Exception:
        logger.exception("failed to initialize DB")

    def _device_loop(device_conf: Dict[str, Any]) -> None:
        rfid_cfg = dict(device_conf.get("rfid") or {})
        rfid_host = str(rfid_cfg.get("host") or "").strip()
        rfid_port = _coerce_int(rfid_cfg.get("port"))
        rfid_device_id = _coerce_int(device_conf.get("rfid_device_id"))
        primary_cam_id = _coerce_int(device_conf.get("primary_cam_id"))
        secondary_cam_id = _coerce_int(device_conf.get("secondary_cam_id"))
        device_index = _coerce_int(device_conf.get("index"))

        # Disable camera stream explicitly when a camera is not configured.
        primary_cam = dict(device_conf.get("primary_camera") or {"host": "", "rtsp_url": "", "index": None})
        secondary_cam = dict(device_conf.get("secondary_camera") or {"host": "", "rtsp_url": "", "index": None})

        station = BascolStation(
            primary_cam=primary_cam,
            secondary_cam=secondary_cam,
            rfid_host=rfid_host,
            rfid_port=rfid_port,
            timing={"tag_cooldown_seconds": 0.0},
        )
        station.start()
        logger.info(
            "device started index=%s rfid_device_id=%s primary_cam_id=%s secondary_cam_id=%s",
            device_index,
            rfid_device_id,
            primary_cam_id,
            secondary_cam_id,
        )
        try:
            while not stop_event.is_set():
                try:
                    result = station.process_next(timeout_sec=1.0)
                except Exception:
                    logger.exception("device loop process_next failed index=%s rfid_device_id=%s", device_index, rfid_device_id)
                    time.sleep(1)
                    continue

                tag = str(getattr(result, "tag", "") or "").strip()
                if not tag:
                    continue

                logger.info(
                    "got result: tag=%s success=%s tags=%s rfid_device_id=%s primary_cam_id=%s secondary_cam_id=%s",
                    tag,
                    bool(result.success),
                    result.tags,
                    rfid_device_id,
                    primary_cam_id,
                    secondary_cam_id,
                )
                with session_lock:
                    _persist_bascol_result_with_tag_session(
                        result=result,
                        station_name=name,
                        db_cfg=db_cfg,
                        tag_sessions=tag_sessions,
                        session_timeout_seconds=session_timeout_seconds,
                        logger=logger,
                        station_id=station_id,
                        station_type=station_type,
                        rfid_device_id=rfid_device_id,
                        primary_cam_id=primary_cam_id,
                        secondary_cam_id=secondary_cam_id,
                    )
        finally:
            station.stop()
            logger.info("device stopped index=%s rfid_device_id=%s", device_index, rfid_device_id)

    threads: list[threading.Thread] = []
    for device_conf in device_confs:
        thread = threading.Thread(target=_device_loop, args=(device_conf,), daemon=True)
        thread.start()
        threads.append(thread)

    try:
        while True:
            active_count = sum(1 for t in threads if t.is_alive())
            if active_count == 0:
                logger.error("all bascol device threads stopped")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("received KeyboardInterrupt, stopping")
    finally:
        stop_event.set()
        for thread in threads:
            thread.join(timeout=3)
        logger.info("stopped")


def _sangshekan_worker(conf: Dict[str, Any]) -> None:
    name = conf.get("name") or "sangshekan"
    station_id = _coerce_int(conf.get("id"))
    station_type = conf.get("type") or "sangshekan"
    logger = _setup_logger(name)
    logger.info("starting sangshekan worker")
    stop_event = threading.Event()
    device_confs = _build_sangshekan_rfid_configs(conf)
    if not device_confs:
        logger.error("no RFID devices configured for sangshekan station")
        return

    db_cfg: Dict[str, Any] = {}
    try:
        db_cfg = _load_db_cfg()
        init_db(db_cfg)
    except Exception:
        logger.exception("failed to initialize DB")

    def _device_loop(device_conf: Dict[str, Any]) -> None:
        rfid_device_id = _coerce_int(device_conf.get("rfid_device_id"))
        device_index = _coerce_int(device_conf.get("index"))
        station = SangShekanStation(
            rfid_host=device_conf.get("host"),
            rfid_port=device_conf.get("port"),
        )
        station.start()
        logger.info("device started index=%s rfid_device_id=%s", device_index, rfid_device_id)
        try:
            while not stop_event.is_set():
                try:
                    event = station.read_next_tag(timeout_sec=1.0)
                except Exception:
                    logger.exception("device loop read_next_tag failed index=%s rfid_device_id=%s", device_index, rfid_device_id)
                    time.sleep(1)
                    continue

                if not getattr(event, "success", False) or not str(getattr(event, "tag", "") or "").strip():
                    continue

                logger.info(
                    "got tag event: tag=%s success=%s rfid_device_id=%s",
                    event.tag,
                    event.success,
                    rfid_device_id,
                )
                payload = _build_sangshekan_payload(
                    event,
                    name,
                    station_id=station_id,
                    station_type=station_type,
                    rfid_device_id=rfid_device_id,
                )
                try:
                    store_result(payload, db_cfg)
                except Exception:
                    logger.exception("failed to store payload to DB")
        finally:
            station.stop()
            logger.info("device stopped index=%s rfid_device_id=%s", device_index, rfid_device_id)

    threads: list[threading.Thread] = []
    for device_conf in device_confs:
        thread = threading.Thread(target=_device_loop, args=(device_conf,), daemon=True)
        thread.start()
        threads.append(thread)

    try:
        while True:
            active_count = sum(1 for t in threads if t.is_alive())
            if active_count == 0:
                logger.error("all sangshekan device threads stopped")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("received KeyboardInterrupt, stopping")
    finally:
        stop_event.set()
        for thread in threads:
            thread.join(timeout=3)
        logger.info("stopped")


if __name__ == "__main__":
    # Multiprocess one process per configured station
    stations = load_stations()

    procs: list[multiprocessing.Process] = []

    for st in stations:
        st_type = (st.get("type") or "").lower()
        if st_type == "bascol":
            p = multiprocessing.Process(target=_bascol_worker, args=(st,))
        elif st_type == "sangshekan":
            p = multiprocessing.Process(target=_sangshekan_worker, args=(st,))
        else:
            print(f"Skipping unknown station type: {st_type}")
            continue
        p.start()
        print(f"Started process {p.pid} for station {st.get('name')}")
        procs.append(p)

    try:
        # Wait for child processes
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        print("Main received KeyboardInterrupt, terminating child processes...")
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        for p in procs:
            p.join()
