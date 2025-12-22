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
    "camera": {
        "host": "192.168.1.111",
        "port": 554,
        "username": "user1",
        "password": "Admin123456",
        "path": "/live/0/main",
        "rtsp_url": "",
    },
    "secondary_camera": {
        "host": "192.168.1.173",
        "port": 554,
        "username": "user1",
        "password": "Admin123456",
        "path": "/live/0/main",
        "rtsp_url": "",
    },
    "rfid": {
        "host": "192.168.1.2",
        "port": 6000,
        "reconnect_seconds": 2.0,
        "debounce_seconds": 1.0,
    },
    "capture": {
        "retry_interval_seconds": 0.4,
        "timeout_seconds": 30.0,
        "target_digits": 5,
        "single_shot_per_detection": True,
        "max_attempts": 5,
        "attempt_reset_mode": "every_detection",  # every_detection | per_tag_session
        "secondary_snapshot_mode": "on_success",  # on_success | on_every_detection
    },
    "storage": {
        "output_dir": "submissions",
        "mismatch_severity": "warning",
        "secondary_capture_behavior": "default",
    },
    "paper_detection": {
        "min_area": 2000,
        "max_area_frac": 0.9,
        "adaptive_block_size": 31,
        "adaptive_C": 5,
        "morph_kernel_size": 5,
        "min_angle": 70.0,
        "max_angle": 110.0,
        "min_std_intensity": 15.0,
        "disable_std_filter": False,
        "min_template_score": 1e-6,
        "retr_mode": "external",  # external | list
        "enable_edge_fallback": True,
        "debug": {
            "enabled": False,
            "save_on_fail": True,
            "max_candidates": 5,
            "output_dir": "paper_debug",
        },
    },
}


def _deep_update(dst: dict, src: dict) -> dict:
    for key, value in (src or {}).items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def _missing_defaults(template: dict, data: dict) -> bool:
    """
    Return True if any key in template is missing from data (recursively for dicts).
    """
    if not isinstance(template, dict):
        return False
    if not isinstance(data, dict):
        return True
    for key, value in template.items():
        if key not in data:
            return True
        if isinstance(value, dict):
            if _missing_defaults(value, data.get(key, {})):
                return True
    return False


def _load_app_config() -> dict:
    cfg = json.loads(json.dumps(DEFAULT_APP_CONFIG))

    raw = None
    if APP_CONFIG_PATH.exists():
        try:
            raw = json.loads(APP_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            raw = None

    if isinstance(raw, dict):
        _deep_update(cfg, raw)

    needs_write = _missing_defaults(DEFAULT_APP_CONFIG, raw if isinstance(raw, dict) else {}) or not APP_CONFIG_PATH.exists()

    if needs_write:
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

mask_path = _resolve_path(APP_CONFIG.get("mask_path", "Mask.png"))
rtsp_url = _env_str("RTSP_URL") or _env_str("CAMERA_RTSP_URL") or _build_rtsp_url(APP_CONFIG.get("camera", {}))
CAMERA_ENABLED = bool(rtsp_url)

secondary_camera_cfg = APP_CONFIG.get("secondary_camera") or APP_CONFIG.get("camera2") or {}
secondary_rtsp_url = _env_str("SECONDARY_RTSP_URL") or _build_rtsp_url(secondary_camera_cfg)
SECONDARY_CAMERA_ENABLED = bool(secondary_rtsp_url)

paper_detection_cfg = APP_CONFIG.get("paper_detection", {}) or {}
paper_debug_cfg = paper_detection_cfg.get("debug", {}) or {}

MIN_AREA = float(paper_detection_cfg.get("min_area", 2000))
MAX_AREA_FRAC = float(paper_detection_cfg.get("max_area_frac", 0.9))
ADAPTIVE_BLOCK_SIZE = int(paper_detection_cfg.get("adaptive_block_size", 31) or 31)
if ADAPTIVE_BLOCK_SIZE % 2 == 0:
    ADAPTIVE_BLOCK_SIZE += 1
if ADAPTIVE_BLOCK_SIZE < 3:
    ADAPTIVE_BLOCK_SIZE = 3
ADAPTIVE_C = float(paper_detection_cfg.get("adaptive_C", 5))
MORPH_KERNEL_SIZE = int(paper_detection_cfg.get("morph_kernel_size", 5) or 5)
if MORPH_KERNEL_SIZE < 1:
    MORPH_KERNEL_SIZE = 1
MIN_ANGLE = float(paper_detection_cfg.get("min_angle", 70.0))
MAX_ANGLE = float(paper_detection_cfg.get("max_angle", 110.0))
MIN_STD_INTENSITY = float(paper_detection_cfg.get("min_std_intensity", 15.0))
DISABLE_STD_FILTER = bool(paper_detection_cfg.get("disable_std_filter", False))
MATCH_WIDTH = 300             # mask matching width
MIN_TM_SCORE = float(paper_detection_cfg.get("min_template_score", 1e-6))
RETR_MODE = str(paper_detection_cfg.get("retr_mode", "external") or "external").strip().lower()
EDGE_FALLBACK_ENABLED = bool(paper_detection_cfg.get("enable_edge_fallback", True))
CONTOUR_RETR_MODE = cv2.RETR_EXTERNAL if RETR_MODE == "external" else cv2.RETR_LIST

PAPER_DEBUG_ENABLED = bool(paper_debug_cfg.get("enabled", False))
PAPER_DEBUG_SAVE_ON_FAIL = bool(paper_debug_cfg.get("save_on_fail", True))
PAPER_DEBUG_MAX_CANDIDATES = int(paper_debug_cfg.get("max_candidates", 5) or 5)
PAPER_DEBUG_DIR = Path(_resolve_path(paper_debug_cfg.get("output_dir", "paper_debug")))
if PAPER_DEBUG_ENABLED:
    try:
        PAPER_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        PAPER_DEBUG_ENABLED = False
        PAPER_DEBUG_SAVE_ON_FAIL = False

COLOR_STD_MIN = MIN_STD_INTENSITY  # legacy compatibility for downstream refs

# OCR / Number Detection Settings
OCR_MODEL_ID          = "./trocr-large-printed"  # local model dir
OCR_CENTER_CROP_RATIO = 0.8
OCR_BAND_HEIGHT_RATIO = 0.5
OCR_DIGITS_ONLY       = True

# Approved RFID tags storage
APPROVED_TAGS_PATH = BASE_DIR / "approved_tags.json"
TAG_LABELS_PATH = BASE_DIR / "tag_labels.json"

rfid_cfg = APP_CONFIG.get("rfid", {})
capture_cfg = APP_CONFIG.get("capture", {})
storage_cfg = APP_CONFIG.get("storage", {})

# RFID reader connection (configure in config.json or via env vars)
RFID_HOST = _env_str("RFID_HOST") or str(rfid_cfg.get("host") or "").strip()
RFID_PORT = _env_int("RFID_PORT") or int(rfid_cfg.get("port") or 6000)
RFID_RECONNECT_SECONDS = _env_float("RFID_RECONNECT_SECONDS") or float(rfid_cfg.get("reconnect_seconds") or 2.0)
RFID_DEBOUNCE_SECONDS = _env_float("RFID_DEBOUNCE_SECONDS") or float(rfid_cfg.get("debounce_seconds") or 1.0)
RFID_QUEUE_MAXSIZE = int(_env_int("RFID_QUEUE_MAXSIZE") or 50)
TAG_COOLDOWN_SECONDS = float(os.environ.get("TAG_COOLDOWN_SECONDS", "3.0"))
TAG_SUBMIT_COOLDOWN_SECONDS = float(os.environ.get("TAG_SUBMIT_COOLDOWN_SECONDS", "300.0"))  # 5 minutes after submit

# Captured data persistence
CAPTURE_OUTPUT_DIR = Path(_resolve_path(storage_cfg.get("output_dir", "submissions")))
CAPTURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Capture loop tuning (until a 5-digit number is found)
CAPTURE_RETRY_INTERVAL_SECONDS = _env_float("CAPTURE_RETRY_INTERVAL_SECONDS") or float(capture_cfg.get("retry_interval_seconds") or 0.4)
CAPTURE_TIMEOUT_SECONDS = _env_float("CAPTURE_TIMEOUT_SECONDS") or float(capture_cfg.get("timeout_seconds") or 30.0)
TARGET_DIGITS = int(capture_cfg.get("target_digits") or 5)
CAPTURE_SINGLE_SHOT = bool(capture_cfg.get("single_shot_per_detection", True))
CAPTURE_MAX_ATTEMPTS = max(1, int(capture_cfg.get("max_attempts") or 5))
ATTEMPT_RESET_MODE = str(capture_cfg.get("attempt_reset_mode") or "every_detection").strip().lower()
if ATTEMPT_RESET_MODE not in {"every_detection", "per_tag_session"}:
    ATTEMPT_RESET_MODE = "every_detection"
SECONDARY_SNAPSHOT_MODE = str(capture_cfg.get("secondary_snapshot_mode") or "on_success").strip().lower()
if SECONDARY_SNAPSHOT_MODE not in {"on_success", "on_every_detection"}:
    SECONDARY_SNAPSHOT_MODE = "on_success"

approved_tags_lock = threading.Lock()
approved_tags = set()
tag_labels_lock = threading.Lock()
TAG_LABELS_VERSION = 1
tag_labels_data = {"version": TAG_LABELS_VERSION, "tag_to_label": {}}
TAG_LABEL_PATTERN = re.compile(r"^\d{5}$")


def _normalize_tag(tag: str) -> str:
    tag = (tag or "").strip().upper()
    # Keep a canonical hex-only representation for EPC/UID inputs.
    tag = re.sub(r"[^0-9A-F]", "", tag)
    return tag


def _load_approved_tags_from_disk() -> set:
    if not APPROVED_TAGS_PATH.exists():
        return set()

    try:
        data = json.loads(APPROVED_TAGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return set()

    if isinstance(data, list):
        raw_tags = data
    elif isinstance(data, dict):
        raw_tags = data.get("approved_tags") or data.get("tags") or []
    else:
        raw_tags = []

    tags = set()
    for raw_tag in raw_tags:
        if not isinstance(raw_tag, str):
            continue
        normalized = _normalize_tag(raw_tag)
        if normalized:
            tags.add(normalized)
    return tags


def _save_approved_tags_to_disk(tags: set) -> None:
    data = {"approved_tags": sorted(tags)}
    tmp_path = APPROVED_TAGS_PATH.with_suffix(APPROVED_TAGS_PATH.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp_path.replace(APPROVED_TAGS_PATH)


with approved_tags_lock:
    approved_tags.update(_load_approved_tags_from_disk())
    if not APPROVED_TAGS_PATH.exists():
        try:
            _save_approved_tags_to_disk(approved_tags)
        except Exception:
            pass


def _load_tag_labels_from_disk() -> dict:
    data = {"version": TAG_LABELS_VERSION, "tag_to_label": {}}
    if not TAG_LABELS_PATH.exists():
        return data

    try:
        raw = json.loads(TAG_LABELS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return data

    if not isinstance(raw, dict):
        return data

    if isinstance(raw.get("version"), int):
        data["version"] = raw["version"]

    raw_map = raw.get("tag_to_label") or raw.get("labels") or {}
    if isinstance(raw_map, dict):
        clean_map = {}
        for raw_tag, raw_label in raw_map.items():
            norm = _normalize_tag(raw_tag)
            if not norm:
                continue
            label_str = str(raw_label).strip()
            if TAG_LABEL_PATTERN.fullmatch(label_str):
                clean_map[norm] = label_str
        data["tag_to_label"] = clean_map

    return data


def _save_tag_labels_to_disk(data: dict) -> None:
    payload = {
        "version": data.get("version", TAG_LABELS_VERSION),
        "tag_to_label": dict(data.get("tag_to_label") or {}),
    }
    tmp_path = TAG_LABELS_PATH.with_suffix(TAG_LABELS_PATH.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(TAG_LABELS_PATH)


def _get_label_for_tag(tag: str) -> str:
    norm = _normalize_tag(tag)
    if not norm:
        return ""
    with tag_labels_lock:
        return (tag_labels_data.get("tag_to_label") or {}).get(norm, "")


def _set_label_for_tag(tag: str, label: str) -> Tuple[bool, str]:
    norm = _normalize_tag(tag)
    if not norm:
        return False, "Invalid tag"

    label_str = (label or "").strip()
    if not TAG_LABEL_PATTERN.fullmatch(label_str):
        return False, "Invalid label; must be exactly 5 digits"

    with tag_labels_lock:
        tag_labels_data.setdefault("tag_to_label", {})
        tag_labels_data["tag_to_label"][norm] = label_str
        tag_labels_data["version"] = TAG_LABELS_VERSION
        try:
            _save_tag_labels_to_disk(tag_labels_data)
        except Exception as e:
            tag_labels_data["tag_to_label"].pop(norm, None)
            return False, f"Failed to save label: {e}"

    return True, "Label saved"


def _delete_label_for_tag(tag: str) -> Tuple[bool, str]:
    norm = _normalize_tag(tag)
    if not norm:
        return False, "Invalid tag"

    with tag_labels_lock:
        existed = False
        tag_labels_data.setdefault("tag_to_label", {})
        if norm in tag_labels_data["tag_to_label"]:
            existed = True
            tag_labels_data["tag_to_label"].pop(norm, None)
        tag_labels_data["version"] = TAG_LABELS_VERSION
        try:
            _save_tag_labels_to_disk(tag_labels_data)
        except Exception as e:
            return False, f"Failed to save labels: {e}"

    return True, "Removed" if existed else "No label found"


with tag_labels_lock:
    tag_labels_data = _load_tag_labels_from_disk()
    if not isinstance(tag_labels_data.get("tag_to_label"), dict):
        tag_labels_data["tag_to_label"] = {}
    if not TAG_LABELS_PATH.exists():
        try:
            _save_tag_labels_to_disk(tag_labels_data)
        except Exception:
            pass


def _tag_label_map() -> dict:
    with tag_labels_lock:
        return dict(tag_labels_data.get("tag_to_label") or {})


def _approve_tag(tag: str) -> Tuple[bool, str]:
    """
    Add tag to approved list + persist. Returns (added, message).
    """
    if not isinstance(tag, str):
        return False, "Invalid tag"
    norm = _normalize_tag(tag)
    if not norm:
        return False, "Invalid tag"

    with approved_tags_lock:
        if norm in approved_tags:
            return False, "Already approved"
        approved_tags.add(norm)
        try:
            _save_approved_tags_to_disk(approved_tags)
        except Exception as e:
            approved_tags.discard(norm)
            return False, f"Failed to save: {e}"

    return True, "Added"


def _is_tag_approved(tag: str) -> bool:
    with approved_tags_lock:
        return tag in approved_tags


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
    "max_attempts": CAPTURE_MAX_ATTEMPTS,
    "attempt_reset_mode": ATTEMPT_RESET_MODE,
    "message": "",
    "raw_text": "",
    "number": "",
    "number_time": None,
    "expected_number": "",
    "observed_number": "",
    "mismatch": False,
    "mismatch_reason": "",
    "mismatch_message": "",
    "original_b64": None,
    "paper_b64": None,
    "awaiting_submit": False,
    "submitted_path": "",
    "submitted_time": None,
    "submitted_prefix": "",
    "secondary_b64": None,
    "secondary_message": "",
    "last_action": "",
}


def _apply_label_to_active_capture(tag: str, label: str) -> None:
    """
    Update in-memory capture state if a label changes for the currently active tag.
    """
    norm = _normalize_tag(tag)
    if not norm:
        return
    label_str = (label or "").strip()
    with state_lock:
        if capture_status.get("tag") != norm:
            return
        capture_status["expected_number"] = label_str

        observed = capture_status.get("observed_number") or capture_status.get("number") or ""
        if not observed:
            capture_status["mismatch"] = False
            capture_status["mismatch_reason"] = ""
            capture_status["mismatch_message"] = ""
            if not label_str:
                capture_status["state"] = "missing_label"
                capture_status["message"] = "No stored label for this tag"
            return

        mismatch = bool(label_str and observed != label_str)
        capture_status["mismatch"] = mismatch
        capture_status["mismatch_reason"] = "mismatch" if mismatch else ""
        capture_status["mismatch_message"] = f"Expected {label_str}, observed {observed}" if mismatch else ""
        if not label_str:
            capture_status["mismatch_reason"] = "missing_label"
            capture_status["mismatch_message"] = "No stored label for this tag"
            capture_status["state"] = "missing_label"
            capture_status["message"] = capture_status["mismatch_message"]
        elif mismatch:
            capture_status["state"] = "mismatch"
            capture_status["message"] = capture_status["mismatch_message"]
        else:
            if capture_status.get("awaiting_submit"):
                capture_status["state"] = "matched"
                capture_status["message"] = "Captured. Awaiting submit."
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


def _next_attempt_for_tag(tag: str) -> int:
    norm = _normalize_tag(tag)
    with state_lock:
        current_tag = capture_status.get("tag")
        current_attempt = int(capture_status.get("attempt") or 0)
        mode = ATTEMPT_RESET_MODE
    if mode == "per_tag_session" and current_tag == norm:
        return current_attempt + 1
    return 1


def _enqueue_retry(tag: str, reason: str = "operator_retry") -> Tuple[bool, str]:
    norm = _normalize_tag(tag)
    if not norm:
        return False, "Invalid tag"

    next_attempt = _next_attempt_for_tag(norm)
    if ATTEMPT_RESET_MODE == "per_tag_session" and next_attempt > CAPTURE_MAX_ATTEMPTS:
        return False, f"Max attempts reached ({next_attempt - 1}/{CAPTURE_MAX_ATTEMPTS})"

    _purge_tag_from_queue(norm)
    try:
        rfid_event_queue.put_nowait(norm)
    except queue.Full:
        return False, "RFID queue full"

    now = time.time()
    _append_event(
        {
            "type": "retry_requested",
            "time": now,
            "time_iso": _utc_iso(now),
            "tag": norm,
            "attempt": next_attempt,
            "requested_by": reason,
        }
    )
    with state_lock:
        capture_status["last_action"] = "retry"
    return True, "Enqueued"


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
        active_states = {"running", "success", "awaiting_submit", "submitted", "matched", "mismatch", "missing_label"}
        last_tag_active = capture_status.get("tag") == tag and capture_status.get("state") in active_states
        pending_submit = bool(capture_status.get("awaiting_submit"))
        cooldown_until = float(tag_submit_cooldowns.get(tag) or 0.0)

        rfid_status["last_tag"] = tag
        rfid_status["last_tag_time"] = now
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
    while rfid_running:
        sock = None
        try:
            with state_lock:
                rfid_status["enabled"] = True
                rfid_status["host"] = RFID_HOST
                rfid_status["port"] = RFID_PORT
                rfid_status["last_error"] = ""

            sock = listen_only.connect(RFID_HOST, RFID_PORT, timeout=5.0)
            with state_lock:
                rfid_status["connected"] = True
                rfid_status["last_error"] = ""

            buf.clear()

            while rfid_running:
                try:
                    chunk = sock.recv(4096)
                except socket.timeout:
                    continue

                if not chunk:
                    raise ConnectionError("Device closed connection")

                buf.extend(chunk)

                for tag, source in listen_only.tags_from_buffer(buf):
                    _handle_rfid_tag(tag, source)

        except Exception as e:
            with state_lock:
                rfid_status["connected"] = False
                rfid_status["last_error"] = str(e)
            time.sleep(RFID_RECONNECT_SECONDS)
        finally:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass


# ============================
# Paper detection for ONE frame
# ============================


def _save_debug_image(path: Path, img: np.ndarray) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), img)
    except Exception:
        pass


def _annotate_candidates(frame: np.ndarray, candidates: list, max_candidates: int, best_idx: int = -1) -> np.ndarray:
    annotated = frame.copy()
    for idx, cand in enumerate(candidates[:max_candidates]):
        quad = cand.get("quad")
        if quad is None:
            continue
        color = (0, 255, 0) if cand.get("accepted") else (0, 165, 255)
        if idx == best_idx:
            color = (0, 0, 255)
        cv2.drawContours(annotated, [quad.astype(int)], -1, color, 3)
        label = f"{idx+1}:{cand.get('score', -1):.3f}"
        reason = cand.get("reject_reason") or ""
        if reason:
            label += f" {reason}"
        cv2.putText(annotated, label, tuple(quad[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return annotated


def _overlay_text(img: np.ndarray, lines: list) -> np.ndarray:
    out = img.copy()
    y = 20
    for line in lines:
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        y += 18
    return out


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
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C
    )

    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    th_closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

    best_score = -1.0
    best_quad = None
    best_warped = None
    candidate_stats = []

    def process_contours(contours, stage_label: str):
        nonlocal best_score, best_quad, best_warped
        for cnt in contours:
            stats = {"stage": stage_label}
            area = cv2.contourArea(cnt)
            stats["area"] = float(area)
            stats["area_frac"] = float(area / frame_area) if frame_area > 0 else 0.0
            if area < MIN_AREA:
                stats["reject_reason"] = "area_small"
                candidate_stats.append(stats)
                continue
            if MAX_AREA_FRAC > 0 and stats["area_frac"] > MAX_AREA_FRAC:
                stats["reject_reason"] = "area_too_big"
                candidate_stats.append(stats)
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            stats["approx_len"] = len(approx)
            if len(approx) != 4:
                stats["reject_reason"] = "not_quad"
                candidate_stats.append(stats)
                continue

            pts = order_points(approx)
            stats["quad"] = pts

            angle_ok = is_rectangle_by_angles(pts, min_angle=MIN_ANGLE, max_angle=MAX_ANGLE)
            stats["angle_ok"] = angle_ok
            if not angle_ok:
                stats["reject_reason"] = "angle"
                candidate_stats.append(stats)
                continue

            w_edge = np.linalg.norm(pts[1] - pts[0])
            h_edge = np.linalg.norm(pts[3] - pts[0])
            if h_edge == 0:
                stats["reject_reason"] = "degenerate"
                candidate_stats.append(stats)
                continue
            aspect = w_edge / h_edge
            stats["aspect"] = float(aspect)

            if not (0.5 * mask_aspect <= aspect <= 2.0 * mask_aspect):
                stats["reject_reason"] = "aspect"
                candidate_stats.append(stats)
                continue

            M = cv2.getPerspectiveTransform(pts.astype(np.float32), dest_quad)
            warped = cv2.warpPerspective(frame, M, (mw, mh))
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

            mean_val, std_val = cv2.meanStdDev(warped_gray)
            std_intensity = float(std_val[0][0])
            stats["std_intensity"] = std_intensity
            if not DISABLE_STD_FILTER and std_intensity < MIN_STD_INTENSITY:
                stats["reject_reason"] = "std_low"
                candidate_stats.append(stats)
                continue

            result = cv2.matchTemplate(warped_gray, mask_resized, cv2.TM_CCOEFF_NORMED)
            _, score, _, _ = cv2.minMaxLoc(result)
            stats["score"] = float(score)

            if score < MIN_TM_SCORE:
                stats["reject_reason"] = "score_low"
                candidate_stats.append(stats)
                continue

            stats["accepted"] = True
            candidate_stats.append(stats)

            if score > best_score:
                best_score = score
                best_quad = pts
                best_warped = warped

    contours, _ = cv2.findContours(th_closed, CONTOUR_RETR_MODE, cv2.CHAIN_APPROX_SIMPLE)
    process_contours(contours, "adaptive")

    if best_quad is None and EDGE_FALLBACK_ENABLED:
        edges = cv2.Canny(blur_gray, 40, 120)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
        edge_contours, _ = cv2.findContours(edges, CONTOUR_RETR_MODE, cv2.CHAIN_APPROX_SIMPLE)
        process_contours(edge_contours, "edge")

    debug_needed = PAPER_DEBUG_ENABLED and (PAPER_DEBUG_SAVE_ON_FAIL or best_quad is not None)
    if debug_needed:
        ts_label = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        sorted_candidates = sorted(candidate_stats, key=lambda c: c.get("score", -1), reverse=True)
        best_idx = -1
        if best_quad is not None:
            for idx, cand in enumerate(sorted_candidates):
                cq = cand.get("quad")
                if cq is not None and np.allclose(cq, best_quad):
                    best_idx = idx
                    break

        annotated = _annotate_candidates(frame, sorted_candidates, PAPER_DEBUG_MAX_CANDIDATES, best_idx)
        _save_debug_image(PAPER_DEBUG_DIR / f"{ts_label}_gray.jpg", gray)
        _save_debug_image(PAPER_DEBUG_DIR / f"{ts_label}_th.jpg", th)
        _save_debug_image(PAPER_DEBUG_DIR / f"{ts_label}_th_closed.jpg", th_closed)
        _save_debug_image(PAPER_DEBUG_DIR / f"{ts_label}_annotated.jpg", annotated)

        if best_warped is not None:
            lines = [
                f"score={best_score:.4f}",
            ]
            bw_with_text = _overlay_text(best_warped, lines)
            _save_debug_image(PAPER_DEBUG_DIR / f"{ts_label}_best_warp.jpg", bw_with_text)

        reason_counts = {}
        for c in candidate_stats:
            reason = c.get("reject_reason") or ("accepted" if c.get("accepted") else "unknown")
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        print(f"[paper_debug] candidates={len(candidate_stats)} best_score={best_score:.4f} reasons={reason_counts}")
        _append_event(
            {
                "type": "paper_debug",
                "time": time.time(),
                "time_iso": _utc_iso(time.time()),
                "best_score": best_score,
                "candidate_count": len(candidate_stats),
                "reason_counts": reason_counts,
            }
        )

    if best_quad is None or best_score < MIN_TM_SCORE:
        return None, None, -1.0

    return best_quad, best_warped, best_score


# ============================
# Init camera + background reader
# ============================

camera_status = {
    "enabled": CAMERA_ENABLED,
    "rtsp_url": rtsp_url,
    "connected": False,
    "last_error": "",
    "last_frame_time": None,
}

cap = None

latest_frame = None
frame_lock = threading.Lock()
reader_running = True

secondary_camera_status = {
    "enabled": SECONDARY_CAMERA_ENABLED,
    "rtsp_url": secondary_rtsp_url,
    "connected": False,
    "last_error": "",
    "last_frame_time": None,
}

cap_secondary = None
secondary_frame = None
secondary_frame_lock = threading.Lock()
secondary_reader_running = True


def _open_camera():
    global cap

    if not CAMERA_ENABLED:
        return False

    try:
        if cap is not None:
            cap.release()
    except Exception:
        pass

    cap = cv2.VideoCapture(rtsp_url)

    # Try to reduce buffering latency if supported
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    return bool(cap.isOpened())


def rtsp_reader():
    """
    Continuously read frames from RTSP and keep the latest one.
    """
    global latest_frame, cap
    while reader_running:
        if not CAMERA_ENABLED:
            with state_lock:
                camera_status["enabled"] = False
                camera_status["connected"] = False
                camera_status["last_error"] = "Camera disabled (missing RTSP config)"
            time.sleep(0.5)
            continue

        if cap is None or not cap.isOpened():
            ok_open = _open_camera()
            with state_lock:
                camera_status["enabled"] = True
                camera_status["rtsp_url"] = rtsp_url
                camera_status["connected"] = bool(ok_open)
                camera_status["last_error"] = "" if ok_open else f"Failed to open RTSP stream: {rtsp_url}"

            if not ok_open:
                time.sleep(1.0)
                continue

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
                camera_status["last_error"] = ""
                camera_status["last_frame_time"] = now
        else:
            with state_lock:
                camera_status["connected"] = False
                if not camera_status.get("last_error"):
                    camera_status["last_error"] = "Failed to read frame"
            time.sleep(0.1)


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

    cap_secondary = cv2.VideoCapture(secondary_rtsp_url)
    try:
        cap_secondary.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return bool(cap_secondary.isOpened())


def secondary_rtsp_reader():
    """
    Continuously read frames from the secondary RTSP camera.
    No detection; used for context capture after OCR success.
    """
    global secondary_frame, cap_secondary
    while secondary_reader_running:
        if not SECONDARY_CAMERA_ENABLED:
            with state_lock:
                secondary_camera_status["enabled"] = False
                secondary_camera_status["connected"] = False
                secondary_camera_status["last_error"] = "Secondary camera disabled (missing RTSP config)"
            time.sleep(1.0)
            continue

        if cap_secondary is None or not cap_secondary.isOpened():
            ok_open = _open_secondary_camera()
            with state_lock:
                secondary_camera_status["enabled"] = True
                secondary_camera_status["rtsp_url"] = secondary_rtsp_url
                secondary_camera_status["connected"] = bool(ok_open)
                secondary_camera_status["last_error"] = "" if ok_open else f"Failed to open RTSP stream: {secondary_rtsp_url}"

            if not ok_open:
                time.sleep(1.5)
                continue

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
                secondary_camera_status["last_error"] = ""
                secondary_camera_status["last_frame_time"] = now
        else:
            with state_lock:
                secondary_camera_status["connected"] = False
                if not secondary_camera_status.get("last_error"):
                    secondary_camera_status["last_error"] = "Failed to read frame"
            time.sleep(0.2)


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


def _get_latest_frame_for_capture(request_after_ts: float = None, max_wait: float = 0.3):
    """
    Try to fetch the freshest frame, waiting briefly if the current frame predates the RFID event.
    """
    start = time.time()
    last_ts = None
    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        with state_lock:
            last_ts = camera_status.get("last_frame_time")
        if frame is not None:
            if request_after_ts is None or (last_ts and last_ts >= request_after_ts):
                return frame, last_ts
        if time.time() - start >= max_wait:
            return frame, last_ts
        time.sleep(0.03)


def grab_and_process():
    """
    Use the latest frame from RTSP, detect paper, run OCR.
    Returns: (success, message, text, annotated_frame, warped_paper)
    """
    global latest_frame

    if not CAMERA_ENABLED:
        return False, "Camera not configured", None, None, None

    with state_lock:
        detection_ts = rfid_status.get("last_tag_time")

    frame, frame_ts = _get_latest_frame_for_capture(request_after_ts=detection_ts, max_wait=0.35)
    if frame is None:
        cam_err = ""
        with state_lock:
            cam_err = camera_status.get("last_error") or ""
        if cam_err:
            return False, cam_err, None, None, None
        return False, "No frame available yet", None, None, None

    frame_stale = bool(detection_ts and frame_ts and frame_ts < detection_ts)

    quad, warped, score = find_best_candidate_single_frame(frame)
    if quad is None:
        annotated = frame.copy()
        msg = "No paper detected"
        if frame_stale:
            msg += " (frame predates RFID event)"
        return False, msg, None, annotated, None

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

    msg = "OK"
    if frame_stale:
        msg += " (frame predates RFID event)"
    return True, msg, text, annotated, warped


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
        "observed_number": capture.get("observed_number") or capture.get("number") or "",
        "expected_number": capture.get("expected_number") or "",
        "mismatch": bool(capture.get("mismatch")),
        "mismatch_reason": capture.get("mismatch_reason") or "",
        "mismatch_message": capture.get("mismatch_message") or "",
        "state": capture.get("state") or "",
        "raw_text": capture.get("raw_text") or "",
        "attempt": attempt,
        "message": capture.get("message") or "",
        "secondary_message": capture.get("secondary_message") or "",
        "timestamp": ts,
        "timestamp_iso": _utc_iso(ts),
        "files": {},
    }

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


capture_running = True
capture_thread = None


def capture_worker():
    """
    Consume approved RFID tag events and run one capture/OCR attempt per detection.
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
    now = time.time()
    expected_number = _get_label_for_tag(tag)

    if not _is_tag_approved(tag):
        with state_lock:
            capture_status.update(
                {
                    "state": "failed",
                    "tag": tag,
                    "started_time": now,
                    "attempt": 0,
                    "max_attempts": CAPTURE_MAX_ATTEMPTS,
                    "attempt_reset_mode": ATTEMPT_RESET_MODE,
                    "message": "Tag not approved",
                    "raw_text": "",
                    "number": "",
                    "number_time": None,
                    "observed_number": "",
                    "expected_number": expected_number,
                    "mismatch": False,
                    "mismatch_reason": "tag_not_approved",
                    "mismatch_message": "Tag not approved",
                    "original_b64": None,
                    "paper_b64": None,
                    "awaiting_submit": False,
                    "submitted_path": "",
                    "submitted_time": None,
                    "submitted_prefix": "",
                    "secondary_b64": None,
                    "secondary_message": "",
                }
            )
        _append_event(
            {
                "type": "tag_not_approved",
                "time": now,
                "time_iso": _utc_iso(now),
                "tag": tag,
            }
        )
        return

    with state_lock:
        prev_tag = capture_status.get("tag")
        prev_attempt = int(capture_status.get("attempt") or 0)
        prev_started = capture_status.get("started_time") or now

    if ATTEMPT_RESET_MODE == "per_tag_session" and prev_tag == tag:
        attempt_base = prev_attempt
        started_time = prev_started
    else:
        attempt_base = 0
        started_time = now

    if attempt_base >= CAPTURE_MAX_ATTEMPTS:
        with state_lock:
            capture_status.update(
                {
                    "state": "failed",
                    "tag": tag,
                    "started_time": started_time,
                    "attempt": attempt_base,
                    "max_attempts": CAPTURE_MAX_ATTEMPTS,
                    "attempt_reset_mode": ATTEMPT_RESET_MODE,
                    "message": f"Max attempts reached ({attempt_base}/{CAPTURE_MAX_ATTEMPTS})",
                    "raw_text": "",
                    "number": "",
                    "number_time": None,
                    "observed_number": "",
                    "expected_number": expected_number,
                    "mismatch": False,
                    "mismatch_reason": "max_attempts",
                    "mismatch_message": "Max attempts reached",
                    "original_b64": None,
                    "paper_b64": None,
                    "awaiting_submit": False,
                    "submitted_path": "",
                    "submitted_time": None,
                    "submitted_prefix": "",
                    "secondary_b64": None,
                    "secondary_message": "",
                }
            )
        _append_event(
            {
                "type": "max_attempts",
                "time": now,
                "time_iso": _utc_iso(now),
                "tag": tag,
                "attempt": attempt_base,
            }
        )
        return

    attempt = attempt_base + 1

    with state_lock:
        capture_status.update(
            {
                "state": "running",
                "tag": tag,
                "started_time": started_time,
                "attempt": attempt,
                "max_attempts": CAPTURE_MAX_ATTEMPTS,
                "attempt_reset_mode": ATTEMPT_RESET_MODE,
                "message": "Capturing...",
                "raw_text": "",
                "number": "",
                "number_time": None,
                "observed_number": "",
                "expected_number": expected_number,
                "mismatch": False,
                "mismatch_reason": "",
                "mismatch_message": "",
                "original_b64": None,
                "paper_b64": None,
                "awaiting_submit": False,
                "submitted_path": "",
                "submitted_time": None,
                "submitted_prefix": "",
                "secondary_b64": None,
                "secondary_message": "",
            }
        )

    success, message, text, annotated, warped = grab_and_process()
    observed_number = _extract_target_digits(text) if success else ""

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

    sec_b64 = None
    sec_msg = ""
    if SECONDARY_SNAPSHOT_MODE == "on_every_detection" or (SECONDARY_SNAPSHOT_MODE == "on_success" and observed_number):
        sec_b64, sec_msg = grab_secondary_snapshot()

    state = "running"
    awaiting_submit = False
    mismatch = False
    mismatch_reason = ""
    mismatch_message = ""
    message_out = message
    number_time = None
    number = ""

    if not success:
        state = "no_paper"
        message_out = message or "No paper detected"
    elif not observed_number:
        state = "ocr_failed"
        message_out = "OCR did not produce a 5-digit number"
    else:
        number_time = time.time()
        number = observed_number
        awaiting_submit = True
        state = "matched"
        message_out = "Captured. Awaiting submit."
        if not expected_number:
            state = "missing_label"
            mismatch_reason = "missing_label"
            mismatch_message = "No stored label for this tag"
            message_out = mismatch_message
        elif observed_number != expected_number:
            state = "mismatch"
            mismatch = True
            mismatch_reason = "mismatch"
            mismatch_message = f"Expected {expected_number}, observed {observed_number}"
            message_out = mismatch_message

    if not awaiting_submit and attempt >= CAPTURE_MAX_ATTEMPTS:
        state = "failed"
        base_reason = mismatch_message or message_out
        mismatch_reason = mismatch_reason or "max_attempts"
        mismatch_message = base_reason + f" (max attempts reached {attempt}/{CAPTURE_MAX_ATTEMPTS})" if base_reason else f"Max attempts reached ({attempt}/{CAPTURE_MAX_ATTEMPTS})"
        message_out = mismatch_message

    with state_lock:
        capture_status.update(
            {
                "state": state,
                "message": message_out,
                "raw_text": text or "",
                "number": number,
                "number_time": number_time,
                "observed_number": observed_number,
                "expected_number": expected_number,
                "mismatch": mismatch,
                "mismatch_reason": mismatch_reason,
                "mismatch_message": mismatch_message,
                "original_b64": original_b64,
                "paper_b64": paper_b64,
                "awaiting_submit": awaiting_submit,
                "submitted_path": "",
                "submitted_time": None,
                "submitted_prefix": "",
                "secondary_b64": sec_b64,
                "secondary_message": sec_msg or "",
                "attempt": attempt,
                "max_attempts": CAPTURE_MAX_ATTEMPTS,
                "attempt_reset_mode": ATTEMPT_RESET_MODE,
            }
        )

    event_type = "capture_attempt"
    if state in {"matched", "awaiting_submit"}:
        event_type = "matched"
    elif state == "mismatch":
        event_type = "mismatch"
    elif state == "missing_label":
        event_type = "missing_label"
    elif state in {"no_paper", "ocr_failed"}:
        event_type = state
    elif state == "failed":
        event_type = "failed"

    event_time = number_time or time.time()
    _append_event(
        {
            "type": event_type,
            "time": event_time,
            "time_iso": _utc_iso(event_time),
            "tag": tag,
            "expected_number": expected_number,
            "observed_number": observed_number,
            "mismatch": mismatch,
            "mismatch_reason": mismatch_reason,
            "mismatch_message": mismatch_message,
            "attempt": attempt,
            "state": state,
            "message": message_out,
        }
    )

    if awaiting_submit:
        _clear_rfid_queue()


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
    return render_template("index.html")


@app.route("/api/tags", methods=["GET"])
def api_get_tags():
    with approved_tags_lock:
        tags = sorted(approved_tags)
    labels = _tag_label_map()
    entries = [{"tag": t, "label": labels.get(t, "")} for t in tags]
    return jsonify(
        {
            "tags": tags,
            "count": len(tags),
            "tag_labels": labels,
            "entries": entries,
        }
    )


@app.route("/api/tags", methods=["POST"])
def api_add_tag():
    payload = request.get_json(silent=True) or {}
    raw = payload.get("tag", "")
    label = (payload.get("label") or payload.get("expected_number") or "").strip()
    tag = _normalize_tag(raw)
    if not tag:
        return jsonify({"success": False, "message": "Missing tag"}), 400

    if label and not TAG_LABEL_PATTERN.fullmatch(label):
        return jsonify({"success": False, "message": "Label must be exactly 5 digits"}), 400

    added, msg = _approve_tag(tag)
    if not added and msg == "Already approved":
        pass
    elif not added:
        return jsonify({"success": False, "message": msg}), 500

    if label:
        ok_label, msg_label = _set_label_for_tag(tag, label)
        if not ok_label:
            return jsonify({"success": False, "message": msg_label}), 500
        _apply_label_to_active_capture(tag, label)

    with approved_tags_lock:
        tags = sorted(approved_tags)
    labels = _tag_label_map()

    return jsonify(
        {
            "success": True,
            "tag": tag,
            "label": labels.get(tag, ""),
            "tags": tags,
            "count": len(tags),
            "tag_labels": labels,
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
        approved_tags.discard(tag_norm)
        try:
            _save_approved_tags_to_disk(approved_tags)
        except Exception as e:
            if existed:
                approved_tags.add(tag_norm)
            return jsonify({"success": False, "message": f"Failed to save: {e}"}), 500

        tags = sorted(approved_tags)

    label_removed = False
    label_message = ""
    if _get_label_for_tag(tag_norm):
        ok_label, label_message = _delete_label_for_tag(tag_norm)
        label_removed = ok_label
        if not ok_label:
            return jsonify({"success": False, "message": label_message}), 500
        _apply_label_to_active_capture(tag_norm, "")

    labels = _tag_label_map()

    response = {"success": True, "removed": existed, "tag": tag_norm, "tags": tags, "count": len(tags), "tag_labels": labels}
    if label_message:
        response["label_message"] = label_message
    response["label_removed"] = label_removed
    return jsonify(response)


@app.route("/api/tags/<tag>/label", methods=["PUT", "POST"])
def api_set_tag_label(tag: str):
    tag_norm = _normalize_tag(tag)
    if not tag_norm:
        return jsonify({"success": False, "message": "Invalid tag"}), 400

    payload = request.get_json(silent=True) or {}
    label = (payload.get("label") or payload.get("expected_number") or payload.get("number") or "").strip()
    if not label:
        return jsonify({"success": False, "message": "Missing label"}), 400
    if not TAG_LABEL_PATTERN.fullmatch(label):
        return jsonify({"success": False, "message": "Label must be exactly 5 digits"}), 400

    ok, msg = _set_label_for_tag(tag_norm, label)
    if not ok:
        return jsonify({"success": False, "message": msg}), 500

    _apply_label_to_active_capture(tag_norm, label)
    labels = _tag_label_map()
    return jsonify({"success": True, "tag": tag_norm, "label": label, "tag_labels": labels})


@app.route("/api/tag-label", methods=["POST"])
def api_set_tag_label_body():
    payload = request.get_json(silent=True) or {}
    tag = _normalize_tag(payload.get("tag", ""))
    label = (payload.get("label") or payload.get("expected_number") or payload.get("number") or "").strip()
    if not tag:
        return jsonify({"success": False, "message": "Missing tag"}), 400
    if not label:
        return jsonify({"success": False, "message": "Missing label"}), 400
    if not TAG_LABEL_PATTERN.fullmatch(label):
        return jsonify({"success": False, "message": "Label must be exactly 5 digits"}), 400

    ok, msg = _set_label_for_tag(tag, label)
    if not ok:
        return jsonify({"success": False, "message": msg}), 500
    _apply_label_to_active_capture(tag, label)
    labels = _tag_label_map()
    return jsonify({"success": True, "tag": tag, "label": label, "tag_labels": labels})


@app.route("/api/status", methods=["GET"])
def api_status():
    include_images = request.args.get("include_images", "0").strip().lower() in {"1", "true", "yes", "on"}
    now = time.time()

    with state_lock:
        rfid = dict(rfid_status)
        capture = dict(capture_status)
        secondary_cam = dict(secondary_camera_status)
        events = list(event_log[-10:])
        queue_size = rfid_event_queue.qsize()

    rfid["queue_size"] = queue_size
    rfid["last_tag_approved"] = bool(rfid.get("last_tag") and _is_tag_approved(rfid["last_tag"]))

    with approved_tags_lock:
        approved_count = len(approved_tags)

    if capture.get("tag"):
        cooldown_until = float(tag_submit_cooldowns.get(capture["tag"]) or 0.0)
        if cooldown_until > 0:
            capture["cooldown_until"] = cooldown_until
            capture["cooldown_until_iso"] = _utc_iso(cooldown_until)

    if rfid.get("last_tag_time") is not None:
        rfid["last_tag_time_iso"] = _utc_iso(rfid["last_tag_time"])

    if capture.get("started_time") is not None:
        capture["started_time_iso"] = _utc_iso(capture["started_time"])
    if capture.get("number_time") is not None:
        capture["number_time_iso"] = _utc_iso(capture["number_time"])

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
            "secondary_camera": secondary_cam,
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

    try:
        saved = _save_capture_to_disk(capture)
    except Exception as e:
        return jsonify({"success": False, "message": f"Failed to save: {e}"}), 500

    cooldown_until = None
    tag = capture.get("tag") or ""
    if tag:
        cooldown_until = time.time() + TAG_SUBMIT_COOLDOWN_SECONDS
        tag_submit_cooldowns[tag] = cooldown_until
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
            "tag": tag,
            "number": capture.get("number") or "",
            "path": saved["files"].get("json", ""),
        }
    )

    return jsonify({"success": True, "saved": saved})


@app.route("/api/capture/retry", methods=["POST"])
def api_capture_retry():
    payload = request.get_json(silent=True) or {}
    req_tag = (payload.get("tag") or "").strip()
    reason = (payload.get("reason") or "operator_retry").strip() or "operator_retry"

    with state_lock:
        current_state = capture_status.get("state") or ""
        current_tag = capture_status.get("tag") or ""
        current_attempt = int(capture_status.get("attempt") or 0)

    tag = req_tag or current_tag
    tag_norm = _normalize_tag(tag)
    if not tag_norm:
        return jsonify({"success": False, "message": "No active tag to retry"}), 400

    if current_state == "running":
        return jsonify({"success": False, "message": "Capture already running"}), 409

    next_attempt = _next_attempt_for_tag(tag_norm)
    if ATTEMPT_RESET_MODE == "per_tag_session" and next_attempt > CAPTURE_MAX_ATTEMPTS:
        return jsonify(
            {
                "success": False,
                "error": "max_attempts_reached",
                "attempt": current_attempt,
                "max_attempts": CAPTURE_MAX_ATTEMPTS,
            }
        ), 409

    ok, msg = _enqueue_retry(tag_norm, reason)
    if not ok:
        return jsonify({"success": False, "message": msg}), 409

    return jsonify({"success": True, "tag": tag_norm, "attempt": next_attempt})


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
        debug_enabled = (
            True
            if debug_env is None
            else debug_env.strip().lower() in {"1", "true", "yes", "on"}
        )
        app.run(debug=debug_enabled, use_reloader=False)
    finally:
        cleanup()
