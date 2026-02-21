from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

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
        "username": "admin",
        "password": "",
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
        "username": "admin",
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
        "device": "cuda",
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
        "mysql": {
            "enabled": False,
            "connection_string": ""
        }
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

_ENV_BOOTSTRAPPED = False


def deep_update(dst: dict, src: Optional[dict]) -> dict:
    for key, value in (src or {}).items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def _as_bool(raw: Any, default: bool = False) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return default
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _coerce_env_value(raw: str) -> Any:
    text = str(raw).strip()
    if not text:
        return ""
    lower = text.lower()
    if lower == "null":
        return None
    if lower in {"true", "false", "yes", "no", "on", "off", "1", "0"}:
        return _as_bool(text)
    if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
        try:
            return json.loads(text)
        except Exception:
            pass
    try:
        if "." in text:
            return float(text)
        return int(text)
    except Exception:
        return text


def _set_nested(cfg: dict, path_parts: list[str], value: Any) -> None:
    cursor = cfg
    for part in path_parts[:-1]:
        key = part.lower()
        if not isinstance(cursor.get(key), dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[path_parts[-1].lower()] = value


def _apply_prefixed_env_overrides(cfg: dict, prefix: str = "CFG__") -> None:
    for key, raw_value in os.environ.items():
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix):]
        parts = [p.strip().lower() for p in suffix.split("__") if p.strip()]
        if not parts:
            continue
        _set_nested(cfg, parts, _coerce_env_value(raw_value))


def load_env_file(env_file: Optional[str] = None, override: bool = False) -> Optional[Path]:
    global _ENV_BOOTSTRAPPED
    if _ENV_BOOTSTRAPPED and env_file is None and not override:
        return None

    try:
        from dotenv import load_dotenv
    except Exception:
        _ENV_BOOTSTRAPPED = True
        return None

    chosen = str(env_file or os.environ.get("ENV_FILE") or "").strip()
    if chosen:
        path = Path(chosen)
        if not path.is_absolute():
            path = Path.cwd() / path
    else:
        path = None
        for candidate in [".env", ".env.local", ".env.docker"]:
            candidate_path = Path.cwd() / candidate
            if candidate_path.exists():
                path = candidate_path
                break
        if path is None:
            _ENV_BOOTSTRAPPED = True
            return None

    load_dotenv(dotenv_path=path, override=override)
    _ENV_BOOTSTRAPPED = True
    return path


def get_base_dir(config_path: Optional[str]) -> Path:
    if not config_path:
        return Path.cwd()
    path = Path(config_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path if path.is_dir() else path.parent


def load_config(config_path: Optional[str] = None, overrides: Optional[dict] = None) -> dict:
    load_env_file()
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    _ = config_path

    app_config_json = str(os.environ.get("APP_CONFIG_JSON") or "").strip()
    if app_config_json:
        try:
            raw = json.loads(app_config_json)
            if isinstance(raw, dict):
                deep_update(cfg, raw)
        except Exception:
            pass

    db_cfg = cfg.setdefault("db", {})
    mysql_cfg = db_cfg.setdefault("mysql", {})
    db_path = os.environ.get("DB_PATH")
    if db_path:
        db_cfg["path"] = db_path
    if os.environ.get("MYSQL_ENABLED") is not None:
        mysql_cfg["enabled"] = _as_bool(os.environ.get("MYSQL_ENABLED"))
    if os.environ.get("MYSQL_CONNECTION_STRING") is not None:
        mysql_cfg["connection_string"] = str(os.environ.get("MYSQL_CONNECTION_STRING") or "").strip()

    _apply_prefixed_env_overrides(cfg)

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


def load_stations() -> list[dict]:
    load_env_file()
    stations_json = str(os.environ.get("STATIONS_JSON") or "").strip()
    if stations_json:
        try:
            parsed = json.loads(stations_json)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    return []
