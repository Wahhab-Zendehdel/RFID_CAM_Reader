from __future__ import annotations

import sys
import os

# Add parent directory to path so we can import lib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.bascol_station import BascolStation
from lib.sangshekan_station import SangShekanStation
from lib.common_config import load_config
from lib.common_db import init_db, store_result
import multiprocessing
from typing import Any, Dict
import sqlite3
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler

import json
import cv2
import time
from datetime import datetime


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
        print(f"  ✗ Failed to save {image_type} image: {e}")
        return ""


def _send_websocket_result(payload: dict) -> None:
    url = os.environ.get("WEBSOCKET_URL", "ws://127.0.0.1:2020")
    if not url:
        print("WEBSOCKET_URL not set; skipping websocket send")
        return
    try:
        try:
            # websocket-client
            from websocket import create_connection
            print(f"  Connecting to {url}...")
            ws = create_connection(url, timeout=5)
            print(f"  Sending {len(str(payload))} bytes...")
            ws.send(json.dumps(payload))
            ws.close()
            print(f"✓ Sent result to {url}")
            return
        except Exception as e:
            print(f"  websocket-client failed: {e}")
            pass
        try:
            # fallback: websockets (async) - attempt a sync import will fail if not installed
            import asyncio
            import websockets

            async def _send():
                async with websockets.connect(url) as w:
                    await w.send(json.dumps(payload))

            print(f"  Using websockets library to send to {url}...")
            asyncio.get_event_loop().run_until_complete(_send())
            print(f"✓ Sent result to {url} (websockets)")
            return
        except Exception as e:
            print(f"  websockets failed: {e}")
            pass
    except Exception as exc:
        print(f"✗ Failed to send websocket payload: {exc}")


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
                tags TEXT,
                number TEXT,
                primary_image_url TEXT,
                secondary_image_url TEXT,
                label_image_url TEXT,
                message TEXT,
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
                datetime, station, tags, number,
                primary_image_url, secondary_image_url, label_image_url,
                message, errors
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.get("datetime"),
                payload.get("station"),
                tags_json,
                payload.get("number"),
                payload.get("primary_image_url"),
                payload.get("secondary_image_url"),
                payload.get("label_image_url"),
                payload.get("message"),
                errors_json,
            ),
        )
        conn.commit()
    except Exception as e:
        print(f"✗ Failed to store result in DB: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass

def run_bascol_demo():
    """
    Continuous demo: listen for RFID tags, process each with camera capture,
    and send results via websocket regardless of success/failure.
    Runs indefinitely until stopped.
    """
    # Load DB config and initialize (will verify MySQL connectivity or create sqlite table)
    cfg = load_config("config/db.json")
    db_cfg = cfg.get("db") or {}
    try:
        init_db(db_cfg)
    except Exception as e:
        print(f"✗ DB initialization warning: {e}")

    station = BascolStation(
        primary_cam="192.168.1.3",
        secondary_cam="192.168.1.201",
        rfid_host="192.168.1.2",
        rfid_port=6000,
    )
    station.start()
    print("📍 Bascol Station started. Listening for RFID tags...")
    print("   Running indefinitely. Press Ctrl+C to stop.\n")
    
    try:
        while True:
            try:
                # Process captures continuously
                # timeout_sec=None means wait indefinitely for tags
                print("⏳ Waiting for RFID tag...")
                result = station.process_next(timeout_sec=None)
                
                print("\n" + "=" * 70)
                print(f"📦 Capture Result (attempt {result.attempts}):")
                print(f"   Tags: {result.tags}")
                print(f"   Success: {result.success}")
                print(f"   Number: {result.number}")
                print(f"   Message: {result.message}")
                if result.errors:
                    print(f"   Errors: {result.errors}")
                print("=" * 70)
                
                # Save images and get URLs
                primary_url = _save_image_to_file(result.primary_image, "primary", result.tag)
                secondary_url = _save_image_to_file(result.secondary_image, "secondary", result.tag)
                label_url = _save_image_to_file(result.label_image, "label", result.tag)
                
                if primary_url:
                    print(f"  📸 Primary image: {primary_url}")
                if secondary_url:
                    print(f"  📸 Secondary image: {secondary_url}")
                if label_url:
                    print(f"  📸 Label image: {label_url}")
                
                # Build normalized payload and store to DB
                payload = _save_and_build_bascol_payload(result, "bascol")
                # ensure image URLs from current capture
                payload["primary_image_url"] = primary_url or None
                payload["secondary_image_url"] = secondary_url or None
                payload["label_image_url"] = label_url or None

                # Persist and send
                try:
                    store_result(payload, db_cfg)
                except Exception as e:
                    print(f"✗ Failed to store result: {e}")
                print("\n🚀 Sending results to WebSocket...")
                _send_websocket_result(payload)
                print("✓ Results sent!\n")
                
            except Exception as e:
                print(f"\n✗ Error processing capture: {e}")
                import traceback
                traceback.print_exc()
                continue
                
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping Bascol Station...")
    except Exception as e:
        print(f"\n✗ Fatal error in demo: {e}")
        import traceback
        traceback.print_exc()
    finally:
        station.stop()
        print("✓ Station stopped.")


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
    


def _save_and_build_bascol_payload(result, station_name: str) -> Dict[str, Any]:
    primary_url = _save_image_to_file(result.primary_image, "primary", result.tag or station_name)
    secondary_url = _save_image_to_file(result.secondary_image, "secondary", result.tag or station_name)
    label_url = _save_image_to_file(result.label_image, "label", result.tag or station_name)

    payload = {
        "datetime": datetime.utcnow().isoformat(),
        "station": station_name,
        "tags": result.tags or ([result.tag] if result.tag else []),
        "number": result.number,
        "primary_image_url": primary_url or None,
        "secondary_image_url": secondary_url or None,
        "label_image_url": label_url or None,
        "message": result.message,
        "errors": result.errors or None,
    }
    return payload


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


def _build_sangshekan_payload(event, station_name: str) -> Dict[str, Any]:
    # Only include the clean fields; other fields should remain null for sangshekan
    return {
        "datetime": datetime.utcnow().isoformat(),
        "station": station_name,
        "tags": [event.tag] if getattr(event, "tag", None) else [],
        "number": None,
        "primary_image_url": None,
        "secondary_image_url": None,
        "label_image_url": None,
        "message": getattr(event, "message", getattr(event, "error", "")),
        "errors": getattr(event, "error", None),
    }


def _bascol_worker(conf: Dict[str, Any]) -> None:
    name = conf.get("name") or "bascol"
    logger = _setup_logger(name)
    logger.info("starting bascol worker")
    station = BascolStation(
        primary_cam=conf.get("primary_cam"),
        secondary_cam=conf.get("secondary_cam"),
        rfid_host=conf.get("rfid_host"),
        rfid_port=conf.get("rfid_port"),
    )
    station.start()
    # Ensure DB exists for this process
    # Load DB config for this worker and initialize
    try:
        cfg = load_config(os.environ.get("DB_CONFIG", "config/db.json"))
        db_cfg = cfg.get("db") or {}
        init_db(db_cfg)
    except Exception:
        logger.exception("failed to initialize DB")
    try:
        while True:
            try:
                result = station.process_next(timeout_sec=None)
                logger.info("got result: success=%s tags=%s", result.success, result.tags)
                payload = _save_and_build_bascol_payload(result, name)
                # persist
                try:
                    store_result(payload, db_cfg)
                except Exception:
                    logger.exception("failed to store payload to DB")
                _send_websocket_result(payload)
            except Exception as e:
                logger.exception("error in processing loop: %s", e)
                time.sleep(1)
                continue
    except KeyboardInterrupt:
        logger.info("received KeyboardInterrupt, stopping")
    finally:
        station.stop()
        logger.info("stopped")


def _sangshekan_worker(conf: Dict[str, Any]) -> None:
    name = conf.get("name") or "sangshekan"
    logger = _setup_logger(name)
    logger.info("starting sangshekan worker")
    station = SangShekanStation(
        rfid_host=conf.get("rfid_host"),
        rfid_port=conf.get("rfid_port"),
    )
    station.start()
    # Ensure DB exists for this process
    # Load DB config for this worker and initialize
    try:
        cfg = load_config(os.environ.get("DB_CONFIG", "config/db.json"))
        db_cfg = cfg.get("db") or {}
        init_db(db_cfg)
    except Exception:
        logger.exception("failed to initialize DB")
    try:
        while True:
            try:
                event = station.read_next_tag(timeout_sec=None)
                logger.info("got tag event: tag=%s success=%s", event.tag, event.success)
                payload = _build_sangshekan_payload(event, name)
                try:
                    store_result(payload, db_cfg)
                except Exception:
                    logger.exception("failed to store payload to DB")
                _send_websocket_result(payload)
            except Exception as e:
                logger.exception("error in processing loop: %s", e)
                time.sleep(1)
                continue
    except KeyboardInterrupt:
        logger.info("received KeyboardInterrupt, stopping")
    finally:
        station.stop()
        logger.info("stopped")


def _load_stations(path: str = "config/stations.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("stations.json must contain a list of station configs")
            return data
    except FileNotFoundError:
        print(f"Stations file not found at {path}")
        return []
    except Exception as e:
        print(f"Failed to load stations file: {e}")
        return []


if __name__ == "__main__":
    # Multiprocess one process per configured station
    stations_file = os.environ.get("STATIONS_FILE", "config/stations.json")
    stations = _load_stations(stations_file)

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
