import json
from pathlib import Path
from typing import Optional

DEFAULT_CONFIG = {
    "auto_fill_missing": False,
    "app": {
        "host": "0.0.0.0",
        "port": 5000,
        "debug": False,
    },
    "camera": {
        "index": None,
        "host": "192.168.1.3",
        "port": 554,
        "username": "user1",
        "password": "Admin123456",
        "path": "/live/0/main",
        "rtsp_url": "",
        "width": None,
        "height": None,
        "fps": None,
        "auto_exposure": None,
        "exposure": None,
        "open_timeout_seconds": 5.0,
        "retry_seconds": 1.0,
        "stale_seconds": 5.0,
        "fail_threshold": 5,
    },
    "secondary_camera": {
        "index": None,
        "host": "192.168.1.4",
        "port": 554,
        "username": "user1",
        "password": "",
        "path": "/live/0/main",
        "rtsp_url": "",
        "width": None,
        "height": None,
        "fps": None,
        "auto_exposure": None,
        "exposure": None,
        "open_timeout_seconds": 5.0,
        "retry_seconds": 1.5,
        "stale_seconds": 6.0,
        "fail_threshold": 5,
    },
    "rfid": {
        "host": "192.168.1.2",
        "port": 6000,
        "reconnect_seconds": 2.0,
        "stale_seconds": 6.0,
        "connect_timeout_seconds": 5.0,
        "read_timeout_seconds": 1.0,
        "debounce_seconds": 1.0,
        "present_window_seconds": 2.0,
        "queue_maxsize": 50,
    },
    "timing": {
        "tag_cooldown_seconds": 3.0,
        "tag_submit_cooldown_seconds": 300.0,
    },
    "capture": {
        "retry_interval_seconds": 0.4,
        "timeout_seconds": 30.0,
        "target_digits": 5,
        "require_label_match_for_submit": True,
    },
    "paper_detection": {
        "mask_path": "Mask.png",
        "min_area": 3000,
        "max_area_frac": 0.1,
        "match_width": 300,
        "min_tm_score": 1e-12,
        "color_std_min": 40.0,
    },
    "ocr": {
        "model_id": "./trocr-large-printed",
        "center_crop_ratio": 0.8,
        "band_height_ratio": 0.5,
        "digits_only": True,
    },
    "ui": {
        "beep_on_detection_default": False,
        "status_poll_ms": 1000,
        "poll_interval_ms": 1000,
    },
    "db": {
        "path": "data/app.db",
    },
    "storage": {
        "images_dir": "data/images",
        "logs_dir": "logs",
    },
    "vehicles": {
        "default_tag_slots": 5,
        "default_label_slots": 1,
    },
}


def deep_update(dst: dict, src: Optional[dict]) -> dict:
    for key, value in (src or {}).items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def get_base_dir(config_path: str) -> Path:
    if not config_path:
        return Path.cwd()
    path = Path(config_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path if path.is_dir() else path.parent


def load_config(config_path: str, overrides: Optional[dict] = None) -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))

    if config_path:
        path = Path(config_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    deep_update(cfg, raw)
            except Exception:
                pass

    if overrides:
        deep_update(cfg, overrides)

    return cfg


def resolve_path(path_value: str, base_dir: Path) -> str:
    path = Path(path_value)
    if not path.is_absolute():
        path = base_dir / path
    return str(path)


def build_rtsp_url(camera_cfg: dict) -> str:
    rtsp = str((camera_cfg or {}).get("rtsp_url") or "").strip()
    if rtsp:
        return rtsp

    host = str((camera_cfg or {}).get("host") or "").strip()
    if not host:
        return ""

    port = int((camera_cfg or {}).get("port") or 554)
    username = str((camera_cfg or {}).get("username") or "").strip()
    password = str((camera_cfg or {}).get("password") or "").strip()
    path = str((camera_cfg or {}).get("path") or "").strip()
    if path and not path.startswith("/"):
        path = "/" + path

    auth = ""
    if username or password:
        auth = f"{username}:{password}@"

    return f"rtsp://{auth}{host}:{port}{path}"
