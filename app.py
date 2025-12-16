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
        try:
            raw = json.loads(APP_CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                _deep_update(cfg, raw)
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

mask_path = _resolve_path(APP_CONFIG.get("mask_path", "Mask.png"))
rtsp_url = _env_str("RTSP_URL") or _env_str("CAMERA_RTSP_URL") or _build_rtsp_url(APP_CONFIG.get("camera", {}))
CAMERA_ENABLED = bool(rtsp_url)

MIN_AREA = 3000               # min contour area in FULL resolution
MAX_AREA_FRAC = 0.10          # max area as fraction of full frame
MATCH_WIDTH = 300             # mask matching width
MIN_TM_SCORE = 0.000000000001 # minimal template match score
COLOR_STD_MIN = 40.0          # reject very uniform patches (e.g. blank)

# OCR / Number Detection Settings
OCR_MODEL_ID          = "./trocr-large-printed"  # local model dir
OCR_CENTER_CROP_RATIO = 0.8
OCR_BAND_HEIGHT_RATIO = 0.5
OCR_DIGITS_ONLY       = True

# Approved RFID tags storage
APPROVED_TAGS_PATH = BASE_DIR / "approved_tags.json"

rfid_cfg = APP_CONFIG.get("rfid", {})
capture_cfg = APP_CONFIG.get("capture", {})

# RFID reader connection (configure in config.json or via env vars)
RFID_HOST = _env_str("RFID_HOST") or str(rfid_cfg.get("host") or "").strip()
RFID_PORT = _env_int("RFID_PORT") or int(rfid_cfg.get("port") or 6000)
RFID_RECONNECT_SECONDS = _env_float("RFID_RECONNECT_SECONDS") or float(rfid_cfg.get("reconnect_seconds") or 2.0)
RFID_DEBOUNCE_SECONDS = _env_float("RFID_DEBOUNCE_SECONDS") or float(rfid_cfg.get("debounce_seconds") or 1.0)
RFID_QUEUE_MAXSIZE = int(_env_int("RFID_QUEUE_MAXSIZE") or 50)
TAG_COOLDOWN_SECONDS = float(os.environ.get("TAG_COOLDOWN_SECONDS", "3.0"))

# Capture loop tuning (until a 5-digit number is found)
CAPTURE_RETRY_INTERVAL_SECONDS = _env_float("CAPTURE_RETRY_INTERVAL_SECONDS") or float(capture_cfg.get("retry_interval_seconds") or 0.4)
CAPTURE_TIMEOUT_SECONDS = _env_float("CAPTURE_TIMEOUT_SECONDS") or float(capture_cfg.get("timeout_seconds") or 30.0)
TARGET_DIGITS = int(capture_cfg.get("target_digits") or 5)

approved_tags_lock = threading.Lock()
approved_tags = set()


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
    "state": "idle",  # idle | running | success | failed
    "tag": "",
    "started_time": None,
    "attempt": 0,
    "message": "",
    "raw_text": "",
    "number": "",
    "number_time": None,
    "original_b64": None,
    "paper_b64": None,
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
    model = VisionEncoderDecoderModel.from_pretrained(model_id)
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

    # If we recently processed this tag, skip re-queuing to keep last images/number visible
    with state_lock:
        last_num_time = capture_status.get("number_time") or capture_status.get("started_time") or 0
        last_tag_active = capture_status.get("tag") == tag and capture_status.get("state") in {"running", "success"}
        if last_tag_active and TAG_COOLDOWN_SECONDS > 0 and (now - last_num_time) < TAG_COOLDOWN_SECONDS:
            return

    debounced = False
    with state_lock:
        rfid_status["last_tag"] = tag
        rfid_status["last_tag_time"] = now
        rfid_status["last_tag_source"] = source

        last_seen = rfid_last_seen_by_tag.get(tag)
        if RFID_DEBOUNCE_SECONDS > 0 and last_seen is not None and (now - last_seen) < RFID_DEBOUNCE_SECONDS:
            debounced = True
        else:
            rfid_last_seen_by_tag[tag] = now

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
        added, msg = _approve_tag(tag)
        approved = added or _is_tag_approved(tag)
        _append_event(
            {
                "type": "auto_approve" if added else "approve_skip",
                "time": now,
                "time_iso": _utc_iso(now),
                "tag": tag,
                "message": msg,
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
            done = time.time()
            with state_lock:
                capture_status["state"] = "success"
                capture_status["number"] = number
                capture_status["number_time"] = done
                capture_status["message"] = "OK"

            _append_event(
                {
                    "type": "number",
                    "time": done,
                    "time_iso": _utc_iso(done),
                    "tag": tag,
                    "number": number,
                    "attempt": attempt,
                }
            )
            return

        if CAPTURE_TIMEOUT_SECONDS > 0 and (time.time() - start) >= CAPTURE_TIMEOUT_SECONDS:
            end = time.time()
            with state_lock:
                capture_status["state"] = "failed"
                capture_status["message"] = f"Timed out after {attempt} attempts"
                capture_status["number_time"] = end

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
    return render_template("index.html")


@app.route("/api/tags", methods=["GET"])
def api_get_tags():
    with approved_tags_lock:
        tags = sorted(approved_tags)
    return jsonify({"tags": tags, "count": len(tags)})


@app.route("/api/tags", methods=["POST"])
def api_add_tag():
    payload = request.get_json(silent=True) or {}
    raw = payload.get("tag", "")
    tag = _normalize_tag(raw)
    if not tag:
        return jsonify({"success": False, "message": "Missing tag"}), 400

    added, msg = _approve_tag(tag)
    if not added and msg == "Already approved":
        with approved_tags_lock:
            tags = sorted(approved_tags)
        return jsonify({"success": True, "tag": tag, "tags": tags, "count": len(tags), "message": msg})
    if not added:
        return jsonify({"success": False, "message": msg}), 500

    with approved_tags_lock:
        tags = sorted(approved_tags)

    return jsonify({"success": True, "tag": tag, "tags": tags, "count": len(tags)})


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

    return jsonify({"success": True, "removed": existed, "tag": tag_norm, "tags": tags, "count": len(tags)})


@app.route("/api/status", methods=["GET"])
def api_status():
    include_images = request.args.get("include_images", "0").strip().lower() in {"1", "true", "yes", "on"}
    now = time.time()

    with state_lock:
        rfid = dict(rfid_status)
        capture = dict(capture_status)
        events = list(event_log[-10:])
        queue_size = rfid_event_queue.qsize()

    rfid["queue_size"] = queue_size
    rfid["last_tag_approved"] = bool(rfid.get("last_tag") and _is_tag_approved(rfid["last_tag"]))

    with approved_tags_lock:
        approved_count = len(approved_tags)

    if rfid.get("last_tag_time") is not None:
        rfid["last_tag_time_iso"] = _utc_iso(rfid["last_tag_time"])

    if capture.get("started_time") is not None:
        capture["started_time_iso"] = _utc_iso(capture["started_time"])
    if capture.get("number_time") is not None:
        capture["number_time_iso"] = _utc_iso(capture["number_time"])

    if not include_images:
        capture["original_b64"] = None
        capture["paper_b64"] = None

    return jsonify(
        {
            "server_time": now,
            "server_time_iso": _utc_iso(now),
            "approved_tags_count": approved_count,
            "rfid": rfid,
            "capture": capture,
            "events": events,
        }
    )


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
    global reader_running, rfid_running, capture_running
    reader_running = False
    rfid_running = False
    capture_running = False
    try:
        reader_thread.join(timeout=1.0)
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
