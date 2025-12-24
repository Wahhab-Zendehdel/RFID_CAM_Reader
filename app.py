import base64
import cv2
import numpy as np
import re
import threading
import time
import os
import json
import queue
import socket
from datetime import datetime, timezone

import listen_only
from typing import Tuple

from flask import Flask, jsonify, render_template, request
from pathlib import Path

import torch
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel


BASE_DIR = Path(__file__).resolve().parent
APP_CONFIG_PATH = BASE_DIR / "config.json"

DEFAULT_APP_CONFIG = {
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
}


def _deep_update(dst: dict, src: dict) -> dict:
    for key, value in (src or {}).items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def _load_app_config() -> dict:
    cfg = json.loads(json.dumps(DEFAULT_APP_CONFIG))

    if APP_CONFIG_PATH.exists():
        auto_fill = False
        try:
            raw = json.loads(APP_CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                auto_fill = bool(raw.get("auto_fill_missing"))
                _deep_update(cfg, raw)
        except Exception:
            pass
        if auto_fill:
            try:
                APP_CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            except Exception:
                pass
        return cfg

    try:
        APP_CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception:
        pass

    return cfg


APP_CONFIG = _load_app_config()


def _env_str(name: str) -> str:
    val = os.environ.get(name)
    return val.strip() if val else ""


def _env_int(name: str):
    val = _env_str(name)
    if not val:
        return None
    try:
        return int(val)
    except ValueError:
        return None


def _env_float(name: str):
    val = _env_str(name)
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _resolve_path(p: str) -> str:
    path = Path(p)
    if not path.is_absolute():
        path = BASE_DIR / path
    return str(path)


def _build_rtsp_url(camera_cfg: dict) -> str:
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


# ============================
# Settings / Parameters
# ============================

app_cfg = APP_CONFIG.get("app", {})
ui_cfg = APP_CONFIG.get("ui", {})
paper_cfg = APP_CONFIG.get("paper_detection", {})
ocr_cfg = APP_CONFIG.get("ocr", {})
camera_cfg = APP_CONFIG.get("camera", {})
secondary_camera_cfg = APP_CONFIG.get("secondary_camera") or APP_CONFIG.get("camera2") or {}
rfid_cfg = APP_CONFIG.get("rfid", {})
capture_cfg = APP_CONFIG.get("capture", {})
timing_cfg = APP_CONFIG.get("timing", {})

mask_path = _resolve_path(paper_cfg.get("mask_path") or APP_CONFIG.get("mask_path", "Mask.png"))
rtsp_url = _env_str("RTSP_URL") or _env_str("CAMERA_RTSP_URL") or _build_rtsp_url(camera_cfg)
camera_index = camera_cfg.get("index")
CAMERA_SOURCE = None
if rtsp_url:
    CAMERA_SOURCE = rtsp_url
elif camera_index is not None:
    try:
        CAMERA_SOURCE = int(camera_index)
    except (TypeError, ValueError):
        CAMERA_SOURCE = None
CAMERA_ENABLED = CAMERA_SOURCE is not None

secondary_rtsp_url = _env_str("SECONDARY_RTSP_URL") or _build_rtsp_url(secondary_camera_cfg)
secondary_camera_index = secondary_camera_cfg.get("index")
SECONDARY_CAMERA_SOURCE = None
if secondary_rtsp_url:
    SECONDARY_CAMERA_SOURCE = secondary_rtsp_url
elif secondary_camera_index is not None:
    try:
        SECONDARY_CAMERA_SOURCE = int(secondary_camera_index)
    except (TypeError, ValueError):
        SECONDARY_CAMERA_SOURCE = None
SECONDARY_CAMERA_ENABLED = SECONDARY_CAMERA_SOURCE is not None

MIN_AREA = int(paper_cfg.get("min_area") or 3000)                 # min contour area in FULL resolution
MAX_AREA_FRAC = float(paper_cfg.get("max_area_frac") or 0.10)     # max area as fraction of full frame
MATCH_WIDTH = int(paper_cfg.get("match_width") or 300)            # mask matching width
MIN_TM_SCORE = float(paper_cfg.get("min_tm_score") or 1e-12)       # minimal template match score
COLOR_STD_MIN = float(paper_cfg.get("color_std_min") or 40.0)      # reject very uniform patches (e.g. blank)

# OCR / Number Detection Settings
OCR_MODEL_ID          = str(ocr_cfg.get("model_id") or "./trocr-large-printed")  # local model dir
OCR_CENTER_CROP_RATIO = float(ocr_cfg.get("center_crop_ratio") or 0.8)
OCR_BAND_HEIGHT_RATIO = float(ocr_cfg.get("band_height_ratio") or 0.5)
OCR_DIGITS_ONLY       = bool(ocr_cfg.get("digits_only", True))

# Approved RFID tags storage
APPROVED_TAGS_PATH = BASE_DIR / "approved_tags.json"

# RFID reader connection (configure in config.json or via env vars)
RFID_HOST = _env_str("RFID_HOST") or str(rfid_cfg.get("host") or "").strip()
RFID_PORT = _env_int("RFID_PORT") or int(rfid_cfg.get("port") or 6000)
RFID_RECONNECT_SECONDS = _env_float("RFID_RECONNECT_SECONDS") or float(rfid_cfg.get("reconnect_seconds") or 2.0)
RFID_STALE_SECONDS = _env_float("RFID_STALE_SECONDS") or float(rfid_cfg.get("stale_seconds") or 0.0)
RFID_CONNECT_TIMEOUT_SECONDS = (
    _env_float("RFID_CONNECT_TIMEOUT_SECONDS") or float(rfid_cfg.get("connect_timeout_seconds") or 5.0)
)
RFID_READ_TIMEOUT_SECONDS = (
    _env_float("RFID_READ_TIMEOUT_SECONDS") or float(rfid_cfg.get("read_timeout_seconds") or 1.0)
)
RFID_DEBOUNCE_SECONDS = _env_float("RFID_DEBOUNCE_SECONDS") or float(rfid_cfg.get("debounce_seconds") or 1.0)
RFID_QUEUE_MAXSIZE = int(_env_int("RFID_QUEUE_MAXSIZE") or (rfid_cfg.get("queue_maxsize") or 50))
RFID_PRESENCE_WINDOW_SECONDS = float(
    _env_float("RFID_PRESENCE_WINDOW_SECONDS") or float(rfid_cfg.get("present_window_seconds") or 2.0)
)
CAMERA_RETRY_SECONDS = float(camera_cfg.get("retry_seconds") or 1.0)
CAMERA_STALE_SECONDS = float(camera_cfg.get("stale_seconds") or 0.0)
CAMERA_FAIL_THRESHOLD = int(camera_cfg.get("fail_threshold") or 5)
SECONDARY_CAMERA_RETRY_SECONDS = float(secondary_camera_cfg.get("retry_seconds") or CAMERA_RETRY_SECONDS)
SECONDARY_CAMERA_STALE_SECONDS = float(secondary_camera_cfg.get("stale_seconds") or CAMERA_STALE_SECONDS)
SECONDARY_CAMERA_FAIL_THRESHOLD = int(secondary_camera_cfg.get("fail_threshold") or CAMERA_FAIL_THRESHOLD)
TAG_COOLDOWN_SECONDS = float(os.environ.get("TAG_COOLDOWN_SECONDS", str(timing_cfg.get("tag_cooldown_seconds") or 3.0)))
TAG_SUBMIT_COOLDOWN_SECONDS = float(
    os.environ.get("TAG_SUBMIT_COOLDOWN_SECONDS", str(timing_cfg.get("tag_submit_cooldown_seconds") or 300.0))
)  # 5 minutes after submit

# Captured data persistence
CAPTURE_OUTPUT_DIR = BASE_DIR / "submissions"
CAPTURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Capture loop tuning (until a 5-digit number is found)
CAPTURE_RETRY_INTERVAL_SECONDS = _env_float("CAPTURE_RETRY_INTERVAL_SECONDS") or float(capture_cfg.get("retry_interval_seconds") or 0.4)
CAPTURE_TIMEOUT_SECONDS = _env_float("CAPTURE_TIMEOUT_SECONDS") or float(capture_cfg.get("timeout_seconds") or 30.0)
TARGET_DIGITS = int(capture_cfg.get("target_digits") or 5)
REQUIRE_LABEL_MATCH_FOR_SUBMIT = bool(capture_cfg.get("require_label_match_for_submit", True))

UI_POLL_INTERVAL_MS = int(ui_cfg.get("status_poll_ms") or ui_cfg.get("poll_interval_ms") or 1000)
UI_BEEP_ON_DETECTION_DEFAULT = bool(ui_cfg.get("beep_on_detection_default", False))

approved_tags_lock = threading.Lock()
approved_tags = {}


def _normalize_tag(tag: str) -> str:
    tag = (tag or "").strip().upper()
    # Keep a canonical hex-only representation for EPC/UID inputs.
    tag = re.sub(r"[^0-9A-F]", "", tag)
    return tag


def _normalize_label_value(label) -> str:
    if label is None:
        return ""
    label_text = str(label).strip()
    if not label_text:
        return ""
    return "".join(re.findall(r"\d", label_text))


def _validate_label_input(label: str) -> Tuple[bool, str, str]:
    if label is None:
        return True, "", ""
    label_text = str(label).strip()
    if not label_text:
        return True, "", ""
    if not re.fullmatch(r"\d+", label_text):
        return False, "", "Label must contain digits only"
    if TARGET_DIGITS > 0 and len(label_text) != TARGET_DIGITS:
        return False, "", f"Label must be {TARGET_DIGITS} digits"
    return True, label_text, ""


def _is_new_tag_format(data) -> bool:
    if not isinstance(data, dict):
        return False
    approved = data.get("approved_tags")
    if not isinstance(approved, dict):
        return False
    for val in approved.values():
        if not isinstance(val, dict):
            return False
    return True


def _load_approved_tags_from_disk() -> Tuple[dict, bool]:
    if not APPROVED_TAGS_PATH.exists():
        return {}, False

    try:
        data = json.loads(APPROVED_TAGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}, False

    needs_migration = not _is_new_tag_format(data)

    tags = {}

    def add_tag_entry(raw_tag, raw_label=""):
        if not isinstance(raw_tag, str):
            return
        normalized = _normalize_tag(raw_tag)
        if not normalized:
            return
        tags[normalized] = {"label": _normalize_label_value(raw_label)}

    def ingest_list(raw_list):
        for item in raw_list:
            if isinstance(item, str):
                add_tag_entry(item, "")
            elif isinstance(item, dict):
                tag_val = item.get("tag")
                label_val = item.get("label", "")
                if tag_val:
                    add_tag_entry(tag_val, label_val)

    def ingest_mapping(raw_map):
        for raw_tag, info in raw_map.items():
            if isinstance(info, dict):
                label_val = info.get("label", "")
            else:
                label_val = info
            add_tag_entry(raw_tag, label_val)

    if isinstance(data, list):
        ingest_list(data)
        return tags, True

    if isinstance(data, dict):
        raw_approved = data.get("approved_tags")
        if isinstance(raw_approved, dict):
            ingest_mapping(raw_approved)
        elif isinstance(raw_approved, list):
            ingest_list(raw_approved)

        raw_tags = data.get("tags")
        if isinstance(raw_tags, list):
            ingest_list(raw_tags)

        if not isinstance(raw_approved, (dict, list)) and not isinstance(raw_tags, list):
            ingest_mapping(data)

    return tags, needs_migration


def _save_approved_tags_to_disk(tags: dict) -> None:
    serialized = {}
    for tag in sorted(tags):
        entry = tags.get(tag) or {}
        label = ""
        if isinstance(entry, dict):
            label = entry.get("label") or ""
        elif isinstance(entry, str):
            label = entry
        serialized[tag] = {"label": _normalize_label_value(label)}
    data = {"approved_tags": serialized}
    tmp_path = APPROVED_TAGS_PATH.with_suffix(APPROVED_TAGS_PATH.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp_path.replace(APPROVED_TAGS_PATH)


with approved_tags_lock:
    loaded_tags, needs_migration = _load_approved_tags_from_disk()
    approved_tags.clear()
    approved_tags.update(loaded_tags)
    if not APPROVED_TAGS_PATH.exists() or needs_migration:
        try:
            _save_approved_tags_to_disk(approved_tags)
        except Exception:
            pass


def _upsert_tag(tag: str, label: str = None) -> Tuple[bool, str]:
    """
    Add or update tag/label + persist. Returns (success, message).
    """
    if not isinstance(tag, str):
        return False, "Invalid tag"
    norm = _normalize_tag(tag)
    if not norm:
        return False, "Invalid tag"

    label_value = None
    if label is not None:
        ok, label_value, err = _validate_label_input(label)
        if not ok:
            return False, err

    with approved_tags_lock:
        entry = approved_tags.get(norm)
        if entry is not None and not isinstance(entry, dict):
            entry = {"label": _normalize_label_value(entry)}
            approved_tags[norm] = entry
        prev_label = entry.get("label") if isinstance(entry, dict) else ""
        changed = False

        if entry is None:
            approved_tags[norm] = {"label": label_value or ""}
            changed = True
        elif label_value is not None and label_value != (prev_label or ""):
            entry["label"] = label_value
            changed = True

        if not changed:
            return True, "Already approved"

        try:
            _save_approved_tags_to_disk(approved_tags)
        except Exception as e:
            if entry is None:
                approved_tags.pop(norm, None)
            elif label_value is not None and isinstance(entry, dict):
                entry["label"] = prev_label or ""
            return False, f"Failed to save: {e}"

    return True, "Updated" if entry is not None else "Added"


def _is_tag_approved(tag: str) -> bool:
    with approved_tags_lock:
        return tag in approved_tags


def _get_tag_label(tag: str) -> str:
    with approved_tags_lock:
        entry = approved_tags.get(tag)
    if isinstance(entry, dict):
        return _normalize_label_value(entry.get("label"))
    if isinstance(entry, str):
        return _normalize_label_value(entry)
    return ""


def _get_approved_tag_items() -> list:
    with approved_tags_lock:
        return [
            {
                "tag": tag,
                "label": _normalize_label_value(entry.get("label") if isinstance(entry, dict) else entry),
            }
            for tag, entry in sorted(approved_tags.items())
        ]


def _find_tags_by_label(label: str) -> list:
    label_norm = _normalize_label_value(label)
    if not label_norm:
        return []
    with approved_tags_lock:
        matches = []
        for tag, entry in approved_tags.items():
            if isinstance(entry, dict):
                entry_label = _normalize_label_value(entry.get("label"))
            else:
                entry_label = _normalize_label_value(entry)
            if entry_label == label_norm:
                matches.append(tag)
        return matches


def _utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


state_lock = threading.Lock()
rfid_event_queue: "queue.Queue[str]" = queue.Queue(maxsize=RFID_QUEUE_MAXSIZE)
rfid_last_seen_by_tag = {}
tag_submit_cooldowns = {}

event_log = []
EVENT_LOG_MAX = 50

rfid_status = {
    "enabled": bool(RFID_HOST),
    "host": RFID_HOST,
    "port": RFID_PORT,
    "connected": False,
    "reconnecting": False,
    "last_ok_ts": None,
    "last_tag_ts": None,
    "consecutive_failures": 0,
    "last_error": "",
    "last_tag": "",
    "last_tag_time": None,
    "last_tag_source": "",
}

capture_status = {
    "state": "idle",  # idle | running | awaiting_submit | submitted | failed
    "tag": "",
    "started_time": None,
    "attempt": 0,
    "message": "",
    "raw_text": "",
    "number": "",
    "number_time": None,
    "original_b64": None,
    "paper_b64": None,
    "awaiting_submit": False,
    "submitted_path": "",
    "submitted_time": None,
    "submitted_prefix": "",
    "secondary_b64": None,
    "secondary_message": "",
    "label_expected": "",
    "label_detected": "",
    "label_match": None,
    "label_message": "",
    "warning_message": "",
    "resolved_tag": "",
    "effective_tag": "",
}


# ============================
# Load mask
# ============================

mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
if mask_img is None:
    raise IOError("Mask not found at: " + mask_path)

mh, mw = mask_img.shape[:2]
scale = MATCH_WIDTH / float(mw)
mask_resized = cv2.resize(mask_img, (MATCH_WIDTH, int(mh * scale)))
mh, mw = mask_resized.shape[:2]
mask_aspect = mw / float(mh)

dest_quad = np.array([[0, 0],
                      [mw - 1, 0],
                      [mw - 1, mh - 1],
                      [0, mh - 1]], dtype=np.float32)


# ============================
# Geometry helpers
# ============================

def order_points(pts):
    """Order contour points as: tl, tr, br, bl."""
    pts = pts.reshape(4, 2)
    pts = sorted(pts, key=lambda x: (x[1], x[0]))  # sort by y, then x
    top = sorted(pts[:2], key=lambda x: x[0])
    bottom = sorted(pts[2:], key=lambda x: x[0])
    return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)


def angle_between(v1, v2):
    """Return angle in degrees between two vectors."""
    v1 = v1.astype(np.float32)
    v2 = v2.astype(np.float32)
    dot = float(np.dot(v1, v2))
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_theta = np.clip(dot / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def is_rectangle_by_angles(pts, min_angle=70.0, max_angle=110.0):
    """
    Check if 4-point polygon is rectangle-like by angle between adjacent edges.
    pts should be ordered (tl, tr, br, bl).
    """
    pts = pts.reshape(4, 2).astype(np.float32)

    edges = [
        pts[1] - pts[0],  # tl->tr
        pts[2] - pts[1],  # tr->br
        pts[3] - pts[2],  # br->bl
        pts[0] - pts[3],  # bl->tl
    ]

    pairs = [
        (edges[3], edges[0]),  # tl
        (edges[0], edges[1]),  # tr
        (edges[1], edges[2]),  # br
        (edges[2], edges[3])   # bl
    ]

    for v_in, v_out in pairs:
        ang = angle_between(v_in, v_out)
        if not (min_angle <= ang <= max_angle):
            return False
    return True


# ============================
# OCR helpers
# ============================

def load_ocr_model(model_id: str):
    image_processor = AutoImageProcessor.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    has_accelerate = False
    try:
        import accelerate  # noqa: F401
        has_accelerate = True
    except Exception:
        has_accelerate = False

    load_kwargs = {"torch_dtype": torch_dtype}
    if has_accelerate:
        load_kwargs["low_cpu_mem_usage"] = True

    try:
        model = VisionEncoderDecoderModel.from_pretrained(model_id, **load_kwargs)
    except TypeError:
        # older transformers may not support low_cpu_mem_usage
        model = VisionEncoderDecoderModel.from_pretrained(model_id, torch_dtype=torch_dtype)
    model.eval()
    return image_processor, tokenizer, model


def _crop_foreground_pil(image: Image.Image) -> Image.Image:
    """
    Try to crop around darker foreground (digits area) automatically.
    """
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray)
    mask = gray.point(lambda p: 255 if p < 200 else 0, mode="1")
    bbox = mask.getbbox()
    if bbox is None:
        return image
    left, top, right, bottom = bbox
    w, h = image.size
    pad_w = int(0.05 * w)
    pad_h = int(0.05 * h)
    left = max(left - pad_w, 0)
    top = max(top - pad_h, 0)
    right = min(right + pad_w, w)
    bottom = min(bottom + pad_h, h)
    return image.crop((left, top, right, bottom))


@torch.inference_mode()
def recognize_text_from_np(
    img_bgr: np.ndarray,
    image_processor,
    tokenizer,
    model,
    center_crop_ratio: float,
    band_height_ratio: float,
    digits_only: bool = True,
    device: torch.device = torch.device("cpu"),
) -> str:
    """
    Run TrOCR on a BGR image (OpenCV) and return recognized text or digits.
    """
    image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    # Optional center crop
    if 0 < center_crop_ratio < 1.0:
        w, h = image.size
        crop_w, crop_h = int(w * center_crop_ratio), int(h * center_crop_ratio)
        left = (w - crop_w) // 2
        top = (h - crop_h) // 2
        image = image.crop((left, top, left + crop_w, top + crop_h))

    # Foreground crop
    image = _crop_foreground_pil(image)

    # Optional horizontal band crop
    if 0 < band_height_ratio < 1.0:
        w, h = image.size
        band_h = int(h * band_height_ratio)
        top = (h - band_h) // 2
        image = image.crop((0, top, w, top + band_h))

    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values.to(device)

    if device.type == "cuda" and next(model.parameters()).dtype == torch.float16:
        pixel_values = pixel_values.half()

    generated_ids = model.generate(
        pixel_values,
        num_beams=3,
        max_length=16,
        early_stopping=True,
    )
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    text = generated_text.strip()

    if digits_only:
        text = "".join(re.findall(r"\d", text))

    return text


# ============================
# RFID helpers + listener
# ============================

# All RFID parsing is handled in `listen_only.py`.


def _append_event(event: dict) -> None:
    with state_lock:
        event_log.append(event)
        if len(event_log) > EVENT_LOG_MAX:
            del event_log[:-EVENT_LOG_MAX]


def _log_event(event_type: str, **fields) -> None:
    now = time.time()
    payload = {"type": event_type, "time": now, "time_iso": _utc_iso(now)}
    payload.update(fields)
    _append_event(payload)


def _clear_rfid_queue() -> int:
    """Drain the RFID event queue (used when waiting for manual submit)."""
    cleared = 0
    try:
        while True:
            rfid_event_queue.get_nowait()
            cleared += 1
    except queue.Empty:
        pass
    return cleared


def _purge_tag_from_queue(tag: str) -> None:
    """Remove all queued occurrences of a tag to avoid flooding."""
    try:
        items = []
        while True:
            items.append(rfid_event_queue.get_nowait())
    except queue.Empty:
        pass
    else:
        for item in items:
            if item != tag:
                try:
                    rfid_event_queue.put_nowait(item)
                except queue.Full:
                    break


def _handle_rfid_tag(raw_tag: str, source: str) -> None:
    tag = _normalize_tag(raw_tag)
    if not tag:
        return

    now = time.time()
    approved = _is_tag_approved(tag)

    debounced = False
    pending_submit = False
    cooldown_until = 0.0

    # If we recently processed this tag, skip re-queuing to keep last images/number visible
    with state_lock:
        last_num_time = capture_status.get("number_time") or capture_status.get("started_time") or 0
        last_tag_active = capture_status.get("tag") == tag and capture_status.get("state") in {"running", "success", "awaiting_submit", "submitted"}
        pending_submit = bool(capture_status.get("awaiting_submit"))
        cooldown_until = float(tag_submit_cooldowns.get(tag) or 0.0)

        rfid_status["last_tag"] = tag
        rfid_status["last_tag_time"] = now
        rfid_status["last_tag_ts"] = now
        rfid_status["last_ok_ts"] = now
        rfid_status["last_tag_source"] = source

        last_seen = rfid_last_seen_by_tag.get(tag)
        if RFID_DEBOUNCE_SECONDS > 0 and last_seen is not None and (now - last_seen) < RFID_DEBOUNCE_SECONDS:
            debounced = True
        else:
            rfid_last_seen_by_tag[tag] = now

    if pending_submit:
        _append_event(
            {
                "type": "blocked",
                "time": now,
                "time_iso": _utc_iso(now),
                "tag": tag,
                "reason": "awaiting_submit",
            }
        )
        return

    if cooldown_until and now < cooldown_until:
        _append_event(
            {
                "type": "cooldown",
                "time": now,
                "time_iso": _utc_iso(now),
                "tag": tag,
                "cooldown_until": cooldown_until,
                "cooldown_until_iso": _utc_iso(cooldown_until),
            }
        )
        return

    if last_tag_active and TAG_COOLDOWN_SECONDS > 0 and (now - last_num_time) < TAG_COOLDOWN_SECONDS:
        return

    if debounced:
        return

    _append_event(
        {
            "type": "rfid",
            "time": now,
            "time_iso": _utc_iso(now),
            "tag": tag,
            "approved": approved,
            "source": source,
        }
    )

    if not approved:
        return

    # Avoid flooding the queue with the same tag if it's already running or queued
    with state_lock:
        current_tag = capture_status.get("tag")
        current_state = capture_status.get("state")
        queued_tags = list(rfid_event_queue.queue)

    if current_state == "running" and current_tag == tag:
        return
    if tag in queued_tags:
        return

    _purge_tag_from_queue(tag)

    try:
        rfid_event_queue.put_nowait(tag)
        _append_event(
            {
                "type": "queue",
                "time": now,
                "time_iso": _utc_iso(now),
                "tag": tag,
            }
        )
    except queue.Full:
        with state_lock:
            rfid_status["last_error"] = "RFID queue full"


rfid_running = True
rfid_thread = None


def rfid_listener():
    """
    Connect to the RFID reader and enqueue approved tag events.
    Enable by setting `rfid.host` in `config.json` (or `RFID_HOST` env var).
    """
    global rfid_running

    if not RFID_HOST:
        return

    buf = bytearray()
    was_connected = False
    was_reconnecting = False
    while rfid_running:
        sock = None
        try:
            with state_lock:
                rfid_status["enabled"] = True
                rfid_status["host"] = RFID_HOST
                rfid_status["port"] = RFID_PORT
                rfid_status["last_error"] = ""
                rfid_status["reconnecting"] = True
            if not was_reconnecting:
                _log_event("rfid_reconnecting", host=RFID_HOST, port=RFID_PORT)
                was_reconnecting = True

            sock = listen_only.connect(RFID_HOST, RFID_PORT, timeout=RFID_CONNECT_TIMEOUT_SECONDS)
            try:
                sock.settimeout(RFID_READ_TIMEOUT_SECONDS)
            except Exception:
                pass
            now = time.time()
            with state_lock:
                rfid_status["connected"] = True
                rfid_status["reconnecting"] = False
                rfid_status["last_error"] = ""
                rfid_status["last_ok_ts"] = now
                rfid_status["consecutive_failures"] = 0
            if not was_connected:
                _log_event("rfid_connected", host=RFID_HOST, port=RFID_PORT)
                was_connected = True
                was_reconnecting = False

            buf.clear()

            while rfid_running:
                try:
                    chunk = sock.recv(4096)
                except socket.timeout:
                    if RFID_STALE_SECONDS:
                        with state_lock:
                            last_ok = rfid_status.get("last_ok_ts") or 0.0
                        if last_ok and (time.time() - last_ok) > RFID_STALE_SECONDS:
                            raise TimeoutError("RFID stale (no data)")
                    continue

                if not chunk:
                    raise ConnectionError("Device closed connection")

                now = time.time()
                with state_lock:
                    rfid_status["last_ok_ts"] = now
                buf.extend(chunk)

                for tag, source in listen_only.tags_from_buffer(buf):
                    _handle_rfid_tag(tag, source)

        except Exception as e:
            err = str(e)
            with state_lock:
                rfid_status["connected"] = False
                rfid_status["reconnecting"] = True
                rfid_status["last_error"] = err
                rfid_status["consecutive_failures"] = int(rfid_status.get("consecutive_failures") or 0) + 1
                failures = rfid_status["consecutive_failures"]
            if was_connected:
                _log_event("rfid_disconnected", error=err)
                was_connected = False
            if not was_reconnecting:
                _log_event("rfid_reconnecting", error=err)
                was_reconnecting = True
            backoff = max(0.2, RFID_RECONNECT_SECONDS * max(1, min(failures, 5)))
            with state_lock:
                rfid_status["last_error"] = err
            time.sleep(backoff)
        finally:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass


# ============================
# Paper detection for ONE frame
# ============================

def find_best_candidate_single_frame(frame):
    """
    Scan a single frame for the best paper candidate.
    Returns (quad, warped_bgr, score) or (None, None, -1.0)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    frame_area = float(h * w)

    blur_gray = cv2.GaussianBlur(gray, (3, 3), 0)

    th = cv2.adaptiveThreshold(
        blur_gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        5
    )

    kernel = np.ones((5, 5), np.uint8)
    th_closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(th_closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best_score = -1.0
    best_quad = None
    best_warped = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue
        if area > MAX_AREA_FRAC * frame_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue

        pts = order_points(approx)

        if not is_rectangle_by_angles(pts, min_angle=70.0, max_angle=110.0):
            continue

        w_edge = np.linalg.norm(pts[1] - pts[0])
        h_edge = np.linalg.norm(pts[3] - pts[0])
        if h_edge == 0:
            continue
        aspect = w_edge / h_edge

        if not (0.5 * mask_aspect <= aspect <= 2.0 * mask_aspect):
            continue

        M = cv2.getPerspectiveTransform(pts.astype(np.float32), dest_quad)
        warped = cv2.warpPerspective(frame, M, (mw, mh))
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        mean_val, std_val = cv2.meanStdDev(warped_gray)
        std_intensity = float(std_val[0][0])
        if std_intensity < COLOR_STD_MIN:
            continue

        result = cv2.matchTemplate(warped_gray, mask_resized, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(result)

        if score > best_score:
            best_score = score
            best_quad = pts
            best_warped = warped

    if best_quad is None or best_score < MIN_TM_SCORE:
        return None, None, -1.0

    return best_quad, best_warped, best_score


# ============================
# Init camera + background reader
# ============================

camera_source_label = rtsp_url or (f"index:{camera_index}" if camera_index is not None else "")
camera_status = {
    "enabled": CAMERA_ENABLED,
    "rtsp_url": camera_source_label,
    "connected": False,
    "reconnecting": False,
    "last_ok_ts": None,
    "last_error": "",
    "last_frame_time": None,
    "last_frame_ts": None,
    "consecutive_failures": 0,
}

cap = None

latest_frame = None
frame_lock = threading.Lock()
reader_running = True

secondary_source_label = secondary_rtsp_url or (f"index:{secondary_camera_index}" if secondary_camera_index is not None else "")
secondary_camera_status = {
    "enabled": SECONDARY_CAMERA_ENABLED,
    "rtsp_url": secondary_source_label,
    "connected": False,
    "reconnecting": False,
    "last_ok_ts": None,
    "last_error": "",
    "last_frame_time": None,
    "last_frame_ts": None,
    "consecutive_failures": 0,
}

cap_secondary = None
secondary_frame = None
secondary_frame_lock = threading.Lock()
secondary_reader_running = True


def _apply_camera_settings(cap, cfg: dict) -> None:
    width = cfg.get("width")
    height = cfg.get("height")
    fps = cfg.get("fps")
    auto_exposure = cfg.get("auto_exposure")
    exposure = cfg.get("exposure")

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


def _open_camera():
    global cap

    if not CAMERA_ENABLED:
        return False

    try:
        if cap is not None:
            cap.release()
    except Exception:
        pass

    cap = cv2.VideoCapture(CAMERA_SOURCE)

    # Try to reduce buffering latency if supported
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    _apply_camera_settings(cap, camera_cfg)

    return bool(cap.isOpened())


def rtsp_reader():
    """
    Continuously read frames from RTSP and keep the latest one.
    """
    global latest_frame, cap
    was_connected = False
    was_reconnecting = False
    while reader_running:
        if not CAMERA_ENABLED:
            with state_lock:
                camera_status["enabled"] = False
                camera_status["connected"] = False
                camera_status["reconnecting"] = False
                camera_status["last_error"] = "Camera disabled (missing camera config)"
            if was_connected:
                _log_event("camera_primary_lost", error="Camera disabled")
                was_connected = False
            time.sleep(0.5)
            continue

        if cap is None or not cap.isOpened():
            with state_lock:
                camera_status["reconnecting"] = True
                camera_status["connected"] = False
            if not was_reconnecting:
                _log_event("camera_primary_reconnecting", error=camera_status.get("last_error") or "")
                was_reconnecting = True
            ok_open = _open_camera()
            now = time.time()
            with state_lock:
                camera_status["enabled"] = True
                camera_status["rtsp_url"] = camera_source_label
                camera_status["connected"] = bool(ok_open)
                camera_status["reconnecting"] = not ok_open
                camera_status["last_error"] = "" if ok_open else f"Failed to open stream: {camera_source_label}"
                if ok_open:
                    camera_status["last_ok_ts"] = now
                    camera_status["consecutive_failures"] = 0

            if not ok_open:
                time.sleep(CAMERA_RETRY_SECONDS)
                continue
            if not was_connected:
                _log_event("camera_primary_connected", source=camera_source_label)
                was_connected = True
                was_reconnecting = False

        try:
            ok, frame = cap.read()
        except Exception as e:
            ok = False
            frame = None
            with state_lock:
                camera_status["connected"] = False
                camera_status["last_error"] = str(e)

        if ok and frame is not None:
            with frame_lock:
                latest_frame = frame
            now = time.time()
            with state_lock:
                camera_status["connected"] = True
                camera_status["reconnecting"] = False
                camera_status["last_error"] = ""
                camera_status["last_frame_time"] = now
                camera_status["last_frame_ts"] = now
                camera_status["last_ok_ts"] = now
                camera_status["consecutive_failures"] = 0
            if not was_connected:
                _log_event("camera_primary_connected", source=camera_source_label)
                was_connected = True
            was_reconnecting = False
        else:
            with state_lock:
                failures = int(camera_status.get("consecutive_failures") or 0) + 1
                camera_status["consecutive_failures"] = failures
                err = camera_status.get("last_error") or "Failed to read frame"
                camera_status["last_error"] = err
            if failures >= CAMERA_FAIL_THRESHOLD:
                try:
                    if cap is not None:
                        cap.release()
                except Exception:
                    pass
                cap = None
                with state_lock:
                    camera_status["connected"] = False
                    camera_status["reconnecting"] = True
                if was_connected:
                    _log_event("camera_primary_lost", error=err)
                    was_connected = False
                if not was_reconnecting:
                    _log_event("camera_primary_reconnecting", error=err)
                    was_reconnecting = True
                time.sleep(CAMERA_RETRY_SECONDS)
                continue
            with state_lock:
                camera_status["connected"] = False
                if not camera_status.get("last_error"):
                    camera_status["last_error"] = "Failed to read frame"
            time.sleep(0.1)

        if CAMERA_STALE_SECONDS:
            with state_lock:
                last_frame_ts = camera_status.get("last_frame_ts")
            if last_frame_ts and (time.time() - last_frame_ts) > CAMERA_STALE_SECONDS:
                try:
                    if cap is not None:
                        cap.release()
                except Exception:
                    pass
                cap = None
                with state_lock:
                    camera_status["connected"] = False
                    camera_status["reconnecting"] = True
                    camera_status["last_error"] = "Camera stale"
                if was_connected:
                    _log_event("camera_primary_lost", error="Camera stale")
                    was_connected = False
                if not was_reconnecting:
                    _log_event("camera_primary_reconnecting", error="Camera stale")
                    was_reconnecting = True
                time.sleep(CAMERA_RETRY_SECONDS)


reader_thread = threading.Thread(target=rtsp_reader, daemon=True)
reader_thread.start()


def _open_secondary_camera():
    global cap_secondary

    if not SECONDARY_CAMERA_ENABLED:
        return False

    try:
        if cap_secondary is not None:
            cap_secondary.release()
    except Exception:
        pass

    cap_secondary = cv2.VideoCapture(SECONDARY_CAMERA_SOURCE)
    try:
        cap_secondary.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    _apply_camera_settings(cap_secondary, secondary_camera_cfg)
    return bool(cap_secondary.isOpened())


def secondary_rtsp_reader():
    """
    Continuously read frames from the secondary RTSP camera.
    No detection; used for context capture after OCR success.
    """
    global secondary_frame, cap_secondary
    was_connected = False
    was_reconnecting = False
    while secondary_reader_running:
        if not SECONDARY_CAMERA_ENABLED:
            with state_lock:
                secondary_camera_status["enabled"] = False
                secondary_camera_status["connected"] = False
                secondary_camera_status["reconnecting"] = False
                secondary_camera_status["last_error"] = "Secondary camera disabled (missing camera config)"
            if was_connected:
                _log_event("camera_secondary_lost", error="Secondary camera disabled")
                was_connected = False
            time.sleep(1.0)
            continue

        if cap_secondary is None or not cap_secondary.isOpened():
            with state_lock:
                secondary_camera_status["reconnecting"] = True
                secondary_camera_status["connected"] = False
            if not was_reconnecting:
                _log_event("camera_secondary_reconnecting", error=secondary_camera_status.get("last_error") or "")
                was_reconnecting = True
            ok_open = _open_secondary_camera()
            now = time.time()
            with state_lock:
                secondary_camera_status["enabled"] = True
                secondary_camera_status["rtsp_url"] = secondary_source_label
                secondary_camera_status["connected"] = bool(ok_open)
                secondary_camera_status["reconnecting"] = not ok_open
                secondary_camera_status["last_error"] = "" if ok_open else f"Failed to open stream: {secondary_source_label}"
                if ok_open:
                    secondary_camera_status["last_ok_ts"] = now
                    secondary_camera_status["consecutive_failures"] = 0

            if not ok_open:
                time.sleep(SECONDARY_CAMERA_RETRY_SECONDS)
                continue
            if not was_connected:
                _log_event("camera_secondary_connected", source=secondary_source_label)
                was_connected = True
                was_reconnecting = False

        try:
            ok, frame = cap_secondary.read()
        except Exception as e:
            ok = False
            frame = None
            with state_lock:
                secondary_camera_status["connected"] = False
                secondary_camera_status["last_error"] = str(e)

        if ok and frame is not None:
            with secondary_frame_lock:
                secondary_frame = frame
            now = time.time()
            with state_lock:
                secondary_camera_status["connected"] = True
                secondary_camera_status["reconnecting"] = False
                secondary_camera_status["last_error"] = ""
                secondary_camera_status["last_frame_time"] = now
                secondary_camera_status["last_frame_ts"] = now
                secondary_camera_status["last_ok_ts"] = now
                secondary_camera_status["consecutive_failures"] = 0
            if not was_connected:
                _log_event("camera_secondary_connected", source=secondary_source_label)
                was_connected = True
            was_reconnecting = False
        else:
            with state_lock:
                failures = int(secondary_camera_status.get("consecutive_failures") or 0) + 1
                secondary_camera_status["consecutive_failures"] = failures
                err = secondary_camera_status.get("last_error") or "Failed to read frame"
                secondary_camera_status["last_error"] = err
            if failures >= SECONDARY_CAMERA_FAIL_THRESHOLD:
                try:
                    if cap_secondary is not None:
                        cap_secondary.release()
                except Exception:
                    pass
                cap_secondary = None
                with state_lock:
                    secondary_camera_status["connected"] = False
                    secondary_camera_status["reconnecting"] = True
                if was_connected:
                    _log_event("camera_secondary_lost", error=err)
                    was_connected = False
                if not was_reconnecting:
                    _log_event("camera_secondary_reconnecting", error=err)
                    was_reconnecting = True
                time.sleep(SECONDARY_CAMERA_RETRY_SECONDS)
                continue
            with state_lock:
                secondary_camera_status["connected"] = False
                if not secondary_camera_status.get("last_error"):
                    secondary_camera_status["last_error"] = "Failed to read frame"
            time.sleep(0.2)

        if SECONDARY_CAMERA_STALE_SECONDS:
            with state_lock:
                last_frame_ts = secondary_camera_status.get("last_frame_ts")
            if last_frame_ts and (time.time() - last_frame_ts) > SECONDARY_CAMERA_STALE_SECONDS:
                try:
                    if cap_secondary is not None:
                        cap_secondary.release()
                except Exception:
                    pass
                cap_secondary = None
                with state_lock:
                    secondary_camera_status["connected"] = False
                    secondary_camera_status["reconnecting"] = True
                    secondary_camera_status["last_error"] = "Secondary camera stale"
                if was_connected:
                    _log_event("camera_secondary_lost", error="Secondary camera stale")
                    was_connected = False
                if not was_reconnecting:
                    _log_event("camera_secondary_reconnecting", error="Secondary camera stale")
                    was_reconnecting = True
                time.sleep(SECONDARY_CAMERA_RETRY_SECONDS)


secondary_reader_thread = threading.Thread(target=secondary_rtsp_reader, daemon=True)
secondary_reader_thread.start()


# ============================
# Init OCR model
# ============================

ocr_image_processor, ocr_tokenizer, ocr_model = load_ocr_model(OCR_MODEL_ID)
ocr_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ocr_model.to(ocr_device)
if ocr_device.type == "cuda":
    ocr_model.half()


# ============================
# Capture + process helper
# ============================

def grab_and_process():
    """
    Use the latest frame from RTSP, detect paper, run OCR.
    Returns: (success, message, text, annotated_frame, warped_paper)
    """
    global latest_frame

    if not CAMERA_ENABLED:
        return False, "Camera not configured", None, None, None

    with frame_lock:
        if latest_frame is None:
            cam_err = ""
            with state_lock:
                cam_err = camera_status.get("last_error") or ""
            if cam_err:
                return False, cam_err, None, None, None
            return False, "No frame available yet", None, None, None
        frame = latest_frame.copy()

    quad, warped, score = find_best_candidate_single_frame(frame)
    if quad is None:
        annotated = frame.copy()
        return False, "No paper detected", None, annotated, None

    text = recognize_text_from_np(
        warped,
        ocr_image_processor,
        ocr_tokenizer,
        ocr_model,
        center_crop_ratio=OCR_CENTER_CROP_RATIO,
        band_height_ratio=OCR_BAND_HEIGHT_RATIO,
        digits_only=OCR_DIGITS_ONLY,
        device=ocr_device,
    )

    annotated = frame.copy()
    cv2.drawContours(annotated, [quad.astype(int)], -1, (0, 255, 0), 3)

    return True, "OK", text, annotated, warped


def img_to_base64_jpeg(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr)
    if not ok:
        raise RuntimeError("Failed to encode image")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def grab_secondary_snapshot():
    """
    Return (b64, message) for latest secondary camera frame.
    Secondary cam is only used for contextual capture; no detection.
    """
    if not SECONDARY_CAMERA_ENABLED:
        return None, "Secondary camera not configured"

    with secondary_frame_lock:
        if secondary_frame is None:
            return None, "Secondary frame not available"
        frame = secondary_frame.copy()

    try:
        return img_to_base64_jpeg(frame), "OK"
    except Exception as e:
        return None, f"Secondary encode failed: {e}"


def _safe_component(text: str, fallback: str = "unknown") -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_-]+", "_", (text or "").strip())
    return cleaned or fallback


def _save_capture_to_disk(capture: dict) -> dict:
    """
    Persist captured data to disk: JSON + images.
    Returns a summary dict with paths and timestamps.
    """
    ts = time.time()
    ts_label = datetime.utcfromtimestamp(ts).strftime("%Y%m%d_%H%M%S")

    tag = _safe_component(capture.get("tag") or "unknown", "tag")
    number = _safe_component(capture.get("number") or "-----", "number")
    attempt = int(capture.get("attempt") or 0)

    prefix = f"{ts_label}_TAG-{tag}_NUM-{number}_ATT-{attempt:02d}"
    json_path = CAPTURE_OUTPUT_DIR / f"{prefix}.json"
    orig_path = CAPTURE_OUTPUT_DIR / f"{prefix}_original.jpg"
    paper_path = CAPTURE_OUTPUT_DIR / f"{prefix}_paper.jpg"
    secondary_path = CAPTURE_OUTPUT_DIR / f"{prefix}_secondary.jpg"

    meta = {
        "tag": capture.get("tag") or "",
        "number": capture.get("number") or "",
        "raw_text": capture.get("raw_text") or "",
        "attempt": attempt,
        "message": capture.get("message") or "",
        "secondary_message": capture.get("secondary_message") or "",
        "timestamp": ts,
        "timestamp_iso": _utc_iso(ts),
        "files": {},
    }
    if capture.get("tag_original"):
        meta["tag_original"] = capture.get("tag_original")
    if capture.get("resolved_tag"):
        meta["resolved_tag"] = capture.get("resolved_tag")

    if capture.get("original_b64"):
        try:
            orig_path.write_bytes(base64.b64decode(capture["original_b64"]))
            meta["files"]["original"] = str(orig_path)
        except Exception:
            meta["files"]["original_error"] = "Failed to save original image"

    if capture.get("paper_b64"):
        try:
            paper_path.write_bytes(base64.b64decode(capture["paper_b64"]))
            meta["files"]["paper"] = str(paper_path)
        except Exception:
            meta["files"]["paper_error"] = "Failed to save paper image"

    if capture.get("secondary_b64"):
        try:
            secondary_path.write_bytes(base64.b64decode(capture["secondary_b64"]))
            meta["files"]["secondary"] = str(secondary_path)
        except Exception:
            meta["files"]["secondary_error"] = "Failed to save secondary image"

    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    meta["files"]["json"] = str(json_path)
    meta["prefix"] = prefix
    return meta


# ============================
# RFID-driven capture worker
# ============================

def _extract_target_digits(text: str) -> str:
    digits = "".join(re.findall(r"\d", text or ""))
    return digits[:TARGET_DIGITS] if len(digits) >= TARGET_DIGITS else ""


def _get_recent_tags(window_seconds: float) -> set:
    now = time.time()
    with state_lock:
        if window_seconds <= 0:
            return set(rfid_last_seen_by_tag.keys())
        return {tag for tag, ts in rfid_last_seen_by_tag.items() if (now - ts) <= window_seconds}


def _resolve_label_for_capture(tag: str, detected_label: str) -> dict:
    detected_label = _normalize_label_value(detected_label)
    expected_label = _normalize_label_value(_get_tag_label(tag))

    label_match = True
    resolved_tag = ""
    warning_message = ""

    if expected_label:
        if detected_label != expected_label:
            now = time.time()
            label_match = False
            warning_message = f"Label mismatch: expected {expected_label}, detected {detected_label}."
            _append_event(
                {
                    "type": "mismatch",
                    "time": now,
                    "time_iso": _utc_iso(now),
                    "tag": tag,
                    "expected": expected_label,
                    "detected": detected_label,
                }
            )

            candidates = _find_tags_by_label(detected_label)
            if candidates:
                present = _get_recent_tags(RFID_PRESENCE_WINDOW_SECONDS)
                candidates = [candidate for candidate in candidates if candidate in present]

            if len(candidates) == 1:
                resolved_tag = candidates[0]
                label_match = True
                warning_message = f"Resolved to tag {resolved_tag} (was {tag})."
                _append_event(
                    {
                        "type": "resolved",
                        "time": now,
                        "time_iso": _utc_iso(now),
                        "tag": tag,
                        "resolved_tag": resolved_tag,
                        "detected_label": detected_label,
                    }
                )
            elif len(candidates) == 0:
                warning_message = (
                    f"Label mismatch: expected {expected_label}, detected {detected_label}. "
                    "No matching tag present."
                )
            else:
                warning_message = (
                    f"Label mismatch: expected {expected_label}, detected {detected_label}. "
                    "Multiple matching tags present."
                )
    else:
        # Allow submit when no expected label is set, but keep a warning visible.
        warning_message = "No expected label set for this tag. Submit allowed."

    effective_tag = resolved_tag or tag

    return {
        "label_expected": expected_label,
        "label_detected": detected_label,
        "label_match": label_match,
        "label_message": warning_message,
        "warning_message": warning_message,
        "resolved_tag": resolved_tag,
        "effective_tag": effective_tag,
    }


capture_running = True
capture_thread = None


def capture_worker():
    """
    Consume approved RFID tag events and run capture/OCR until a 5-digit number is found.
    """
    global capture_running

    while capture_running:
        try:
            tag = rfid_event_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            _process_capture_for_tag(tag)
        finally:
            try:
                rfid_event_queue.task_done()
            except Exception:
                pass


def _process_capture_for_tag(tag: str) -> None:
    if not _is_tag_approved(tag):
        now = time.time()
        with state_lock:
            capture_status.update(
                {
                    "state": "failed",
                    "tag": tag,
                    "started_time": now,
                    "attempt": 0,
                    "message": "Tag not approved",
                    "raw_text": "",
                    "number": "",
                    "number_time": None,
                    "awaiting_submit": False,
                    "submitted_path": "",
                    "submitted_time": None,
                    "submitted_prefix": "",
                    "secondary_b64": None,
                    "secondary_message": "",
                    "label_expected": "",
                    "label_detected": "",
                    "label_match": None,
                    "label_message": "",
                    "resolved_tag": "",
                    "warning_message": "",
                    "effective_tag": "",
                }
            )
        return

    start = time.time()
    with state_lock:
        capture_status.update(
            {
                "state": "running",
                "tag": tag,
                "started_time": start,
                "attempt": 0,
                "message": "Capturing...",
                "raw_text": "",
                "number": "",
                "number_time": None,
                "original_b64": None,
                "paper_b64": None,
                "awaiting_submit": False,
                "submitted_path": "",
                "submitted_time": None,
                "submitted_prefix": "",
                "secondary_b64": None,
                "secondary_message": "",
                "label_expected": "",
                "label_detected": "",
                "label_match": None,
                "label_message": "",
                "resolved_tag": "",
                "warning_message": "",
                "effective_tag": "",
            }
        )

    attempt = 0
    while capture_running:
        attempt += 1
        success, message, text, annotated, warped = grab_and_process()
        number = _extract_target_digits(text) if success else ""

        original_b64 = None
        paper_b64 = None
        if annotated is not None:
            try:
                original_b64 = img_to_base64_jpeg(annotated)
            except Exception:
                original_b64 = None
        if warped is not None:
            try:
                paper_b64 = img_to_base64_jpeg(warped)
            except Exception:
                paper_b64 = None

        with state_lock:
            capture_status["attempt"] = attempt
            capture_status["message"] = message
            capture_status["raw_text"] = text or ""
            capture_status["original_b64"] = original_b64
            capture_status["paper_b64"] = paper_b64

        if number:
            sec_b64, sec_msg = grab_secondary_snapshot()
            done = time.time()
            label_info = _resolve_label_for_capture(tag, number)
            with state_lock:
                capture_status["state"] = "awaiting_submit"
                capture_status["number"] = number
                capture_status["number_time"] = done
                capture_status["message"] = "Captured. Awaiting submit."
                capture_status["awaiting_submit"] = True
                capture_status["submitted_path"] = ""
                capture_status["submitted_time"] = None
                capture_status["submitted_prefix"] = ""
                capture_status["secondary_b64"] = sec_b64
                capture_status["secondary_message"] = sec_msg or ""
                capture_status["label_expected"] = label_info.get("label_expected", "")
                capture_status["label_detected"] = label_info.get("label_detected", "")
                capture_status["label_match"] = label_info.get("label_match")
                capture_status["label_message"] = label_info.get("label_message", "")
                capture_status["resolved_tag"] = label_info.get("resolved_tag", "")
                capture_status["warning_message"] = label_info.get("warning_message", "")
                capture_status["effective_tag"] = label_info.get("effective_tag", "")

            _clear_rfid_queue()
            _append_event(
                {
                    "type": "number",
                    "time": done,
                    "time_iso": _utc_iso(done),
                    "tag": tag,
                    "number": number,
                    "attempt": attempt,
                    "awaiting_submit": True,
                }
            )
            return

        if CAPTURE_TIMEOUT_SECONDS > 0 and (time.time() - start) >= CAPTURE_TIMEOUT_SECONDS:
            end = time.time()
            with state_lock:
                capture_status["state"] = "failed"
                capture_status["message"] = f"Timed out after {attempt} attempts"
                capture_status["number_time"] = end
                capture_status["awaiting_submit"] = False
                capture_status["submitted_path"] = ""
                capture_status["submitted_time"] = None
                capture_status["submitted_prefix"] = ""
                capture_status["label_expected"] = ""
                capture_status["label_detected"] = ""
                capture_status["label_match"] = None
                capture_status["label_message"] = ""
                capture_status["resolved_tag"] = ""
                capture_status["warning_message"] = ""
                capture_status["effective_tag"] = ""

            _append_event(
                {
                    "type": "timeout",
                    "time": end,
                    "time_iso": _utc_iso(end),
                    "tag": tag,
                    "attempt": attempt,
                }
            )
            return

        time.sleep(CAPTURE_RETRY_INTERVAL_SECONDS)


# ============================
# Flask app
# ============================

app = Flask(__name__)

if RFID_HOST:
    rfid_thread = threading.Thread(target=rfid_listener, daemon=True)
    rfid_thread.start()
else:
    with state_lock:
        rfid_status["enabled"] = False

capture_thread = threading.Thread(target=capture_worker, daemon=True)
capture_thread.start()


@app.route("/")
def index():
    return render_template(
        "index.html",
        target_digits=TARGET_DIGITS,
        poll_interval_ms=UI_POLL_INTERVAL_MS,
        beep_default=UI_BEEP_ON_DETECTION_DEFAULT,
        require_label_match=REQUIRE_LABEL_MATCH_FOR_SUBMIT,
        present_window_seconds=RFID_PRESENCE_WINDOW_SECONDS,
    )


@app.route("/api/tags", methods=["GET"])
def api_get_tags():
    items = _get_approved_tag_items()
    return jsonify({"items": items, "count": len(items), "target_digits": TARGET_DIGITS})


@app.route("/api/tags", methods=["POST"])
def api_add_tag():
    payload = request.get_json(silent=True) or {}
    raw = payload.get("tag", "")
    tag = _normalize_tag(raw)
    if not tag:
        return jsonify({"success": False, "message": "Missing tag"}), 400
    label = payload.get("label") if "label" in payload else None

    ok, msg = _upsert_tag(tag, label)
    if not ok:
        status = 400 if ("Label" in msg or "Invalid" in msg) else 500
        return jsonify({"success": False, "message": msg}), status

    items = _get_approved_tag_items()
    return jsonify(
        {
            "success": True,
            "item": {"tag": tag, "label": _get_tag_label(tag)},
            "items": items,
            "count": len(items),
            "message": msg,
        }
    )


@app.route("/api/tags/<tag>", methods=["DELETE"])
def api_delete_tag(tag: str):
    tag_norm = _normalize_tag(tag)
    if not tag_norm:
        return jsonify({"success": False, "message": "Invalid tag"}), 400

    with approved_tags_lock:
        existed = tag_norm in approved_tags
        prev_entry = approved_tags.get(tag_norm)
        if existed:
            approved_tags.pop(tag_norm, None)
        try:
            _save_approved_tags_to_disk(approved_tags)
        except Exception as e:
            if existed:
                approved_tags[tag_norm] = prev_entry
            return jsonify({"success": False, "message": f"Failed to save: {e}"}), 500

    items = _get_approved_tag_items()
    return jsonify({"success": True, "removed": existed, "tag": tag_norm, "items": items, "count": len(items)})


@app.route("/api/status", methods=["GET"])
def api_status():
    include_images = request.args.get("include_images", "0").strip().lower() in {"1", "true", "yes", "on"}
    now = time.time()

    with state_lock:
        rfid = dict(rfid_status)
        capture = dict(capture_status)
        cam_primary = dict(camera_status)
        cam_secondary = dict(secondary_camera_status)
        events = list(event_log[-10:])
        queue_size = rfid_event_queue.qsize()

    rfid["queue_size"] = queue_size
    rfid["last_tag_approved"] = bool(rfid.get("last_tag") and _is_tag_approved(rfid["last_tag"]))

    with approved_tags_lock:
        approved_count = len(approved_tags)

    effective_tag = capture.get("effective_tag") or capture.get("resolved_tag") or capture.get("tag")
    if effective_tag:
        cooldown_until = float(tag_submit_cooldowns.get(effective_tag) or 0.0)
        if cooldown_until > 0:
            capture["cooldown_until"] = cooldown_until
            capture["cooldown_until_iso"] = _utc_iso(cooldown_until)
            capture["cooldown_tag"] = effective_tag

    if rfid.get("last_tag_time") is not None:
        rfid["last_tag_time_iso"] = _utc_iso(rfid["last_tag_time"])
    if rfid.get("last_ok_ts") is not None:
        rfid["last_ok_ts_iso"] = _utc_iso(rfid["last_ok_ts"])
        rfid["seconds_since_last_ok"] = max(0.0, now - float(rfid["last_ok_ts"]))
    last_tag_ts = rfid.get("last_tag_ts") or rfid.get("last_tag_time")
    if last_tag_ts is not None:
        rfid["last_tag_ts"] = last_tag_ts
        rfid["last_tag_ts_iso"] = _utc_iso(last_tag_ts)
        rfid["seconds_since_last_tag"] = max(0.0, now - float(last_tag_ts))
    if RFID_STALE_SECONDS and rfid.get("last_ok_ts"):
        rfid["stale"] = (now - float(rfid["last_ok_ts"])) > RFID_STALE_SECONDS
    else:
        rfid["stale"] = False

    if capture.get("started_time") is not None:
        capture["started_time_iso"] = _utc_iso(capture["started_time"])
    if capture.get("number_time") is not None:
        capture["number_time_iso"] = _utc_iso(capture["number_time"])

    for cam in (cam_primary, cam_secondary):
        if cam.get("last_frame_time") is not None:
            cam["last_frame_time_iso"] = _utc_iso(cam["last_frame_time"])
        if cam.get("last_frame_ts") is not None:
            cam["last_frame_ts_iso"] = _utc_iso(cam["last_frame_ts"])
            cam["seconds_since_last_frame"] = max(0.0, now - float(cam["last_frame_ts"]))

    if CAMERA_STALE_SECONDS and cam_primary.get("last_frame_ts"):
        cam_primary["stale"] = (now - float(cam_primary["last_frame_ts"])) > CAMERA_STALE_SECONDS
    else:
        cam_primary["stale"] = False

    if SECONDARY_CAMERA_STALE_SECONDS and cam_secondary.get("last_frame_ts"):
        cam_secondary["stale"] = (now - float(cam_secondary["last_frame_ts"])) > SECONDARY_CAMERA_STALE_SECONDS
    else:
        cam_secondary["stale"] = False

    if not include_images:
        capture["original_b64"] = None
        capture["paper_b64"] = None
        capture["secondary_b64"] = None

    return jsonify(
        {
            "server_time": now,
            "server_time_iso": _utc_iso(now),
            "approved_tags_count": approved_count,
            "rfid": rfid,
            "cameras": {"primary": cam_primary, "secondary": cam_secondary},
            "secondary_camera": cam_secondary,
            "capture": capture,
            "events": events,
        }
    )


@app.route("/api/capture/submit", methods=["POST"])
def api_submit_capture():
    with state_lock:
        capture = dict(capture_status)

    if not capture.get("awaiting_submit"):
        return jsonify({"success": False, "message": "No capture awaiting submit"}), 400
    if not capture.get("number"):
        return jsonify({"success": False, "message": "No number captured"}), 400
    if REQUIRE_LABEL_MATCH_FOR_SUBMIT and capture.get("label_match") is False:
        message = capture.get("warning_message") or capture.get("label_message") or "Label mismatch; submit blocked"
        return jsonify({"success": False, "message": message}), 400

    effective_tag = capture.get("effective_tag") or capture.get("resolved_tag") or capture.get("tag") or ""
    save_capture = dict(capture)
    if effective_tag:
        if capture.get("tag") and capture.get("tag") != effective_tag:
            save_capture["tag_original"] = capture.get("tag")
        save_capture["tag"] = effective_tag
        save_capture["effective_tag"] = effective_tag

    try:
        saved = _save_capture_to_disk(save_capture)
    except Exception as e:
        return jsonify({"success": False, "message": f"Failed to save: {e}"}), 500

    cooldown_until = None
    if effective_tag:
        cooldown_until = time.time() + TAG_SUBMIT_COOLDOWN_SECONDS
        tag_submit_cooldowns[effective_tag] = cooldown_until
        saved["cooldown_until"] = cooldown_until
        saved["cooldown_until_iso"] = _utc_iso(cooldown_until)

    with state_lock:
        capture_status["awaiting_submit"] = False
        capture_status["state"] = "submitted"
        capture_status["message"] = "Submitted"
        capture_status["submitted_path"] = saved["files"].get("json", "")
        capture_status["submitted_time"] = saved.get("timestamp")
        capture_status["submitted_prefix"] = saved.get("prefix", "")

    _append_event(
        {
            "type": "submitted",
            "time": saved.get("timestamp", time.time()),
            "time_iso": saved.get("timestamp_iso", _utc_iso(time.time())),
            "tag": effective_tag,
            "tag_original": capture.get("tag") if effective_tag and capture.get("tag") != effective_tag else "",
            "number": capture.get("number") or "",
            "label_detected": capture.get("label_detected") or "",
            "label_match": capture.get("label_match"),
            "path": saved["files"].get("json", ""),
        }
    )

    return jsonify({"success": True, "saved": saved})


@app.route("/capture")
def capture():
    success, message, text, annotated, warped = grab_and_process()

    if not success:
        response = {
            "success": False,
            "message": message,
            "text": "",
            "original": None,
            "paper": None,
        }
        if annotated is not None:
            try:
                response["original"] = img_to_base64_jpeg(annotated)
            except Exception:
                pass
        return jsonify(response)

    try:
        orig_b64 = img_to_base64_jpeg(annotated)
        warped_b64 = img_to_base64_jpeg(warped)
    except Exception as e:
        return jsonify(
            {
                "success": False,
                "message": f"Encoding error: {e}",
                "text": "",
                "original": None,
                "paper": None,
            }
        )

    return jsonify(
        {
            "success": True,
            "message": message,
            "text": text or "",
            "original": orig_b64,
            "paper": warped_b64,
        }
    )


# ============================
# Graceful shutdown (optional)
# ============================

def cleanup():
    global reader_running, rfid_running, capture_running, secondary_reader_running
    reader_running = False
    rfid_running = False
    capture_running = False
    secondary_reader_running = False
    try:
        reader_thread.join(timeout=1.0)
    except Exception:
        pass
    try:
        secondary_reader_thread.join(timeout=1.0)
    except Exception:
        pass
    if rfid_thread is not None:
        try:
            rfid_thread.join(timeout=1.0)
        except Exception:
            pass
    if capture_thread is not None:
        try:
            capture_thread.join(timeout=1.0)
        except Exception:
            pass
    try:
        if cap is not None:
            cap.release()
    except Exception:
        pass
    try:
        if cap_secondary is not None:
            cap_secondary.release()
    except Exception:
        pass


if __name__ == "__main__":
    try:
        # Flask's debug reloader runs the script twice on Windows (spawning a second process),
        # which can easily double memory usage with large ML models loaded at import time.
        debug_env = os.environ.get("FLASK_DEBUG") or os.environ.get("DEBUG")
        if debug_env is None:
            debug_enabled = bool(app_cfg.get("debug", False))
        else:
            debug_enabled = debug_env.strip().lower() in {"1", "true", "yes", "on"}
        app_host = _env_str("APP_HOST") or str(app_cfg.get("host") or "0.0.0.0")
        app_port = _env_int("APP_PORT") or int(app_cfg.get("port") or 5000)
        app.run(host=app_host, port=app_port, debug=debug_enabled, use_reloader=False)
    finally:
        cleanup()
