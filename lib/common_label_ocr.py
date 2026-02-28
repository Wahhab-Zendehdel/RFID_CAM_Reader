from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

from .common_config import resolve_path

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency at import time
    YOLO = None


class LabelDetector:
    def __init__(self, paper_cfg: dict, ocr_cfg: dict, base_dir: Optional[Path] = None) -> None:
        self.paper_cfg = paper_cfg or {}
        self.ocr_cfg = ocr_cfg or {}
        self.base_dir = base_dir or Path.cwd()

        self.detector_model_path = self._resolve_detector_model_path()
        self.detector_conf = float(self.paper_cfg.get("confidence") or self.paper_cfg.get("conf") or 0.5)
        self.detector = self._load_detector(self.detector_model_path)

        self.ocr_model_id = str(self.ocr_cfg.get("model_id") or "./trocr-large-printed")
        self.ocr_center_crop_ratio = float(self.ocr_cfg.get("center_crop_ratio") or 0.8)
        self.ocr_band_height_ratio = float(self.ocr_cfg.get("band_height_ratio") or 0.5)
        self.ocr_digits_only = bool(self.ocr_cfg.get("digits_only", True))

        model_id = self._resolve_model_id(self.ocr_model_id)
        self.ocr_image_processor, self.ocr_tokenizer, self.ocr_model = self._load_ocr_model(model_id)

        requested = str(self.ocr_cfg.get("device") or "").strip().lower()
        if requested in ("cpu", "cuda"):
            if requested == "cuda" and not torch.cuda.is_available():
                self.ocr_device = torch.device("cpu")
            else:
                self.ocr_device = torch.device(requested)
        else:
            self.ocr_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ocr_model.to(self.ocr_device)
        self.ocr_model.eval()
        if self.ocr_device.type == "cuda":
            try:
                self.ocr_model.half()
            except Exception:
                pass

    def _resolve_detector_model_path(self) -> str:
        raw = (
            self.paper_cfg.get("model_path")
            or self.paper_cfg.get("yolo_model_path")
            or self.paper_cfg.get("weights_path")
            or "paper_dataset_aug/runs/detect/train/weights/best.pt"
        )
        primary = Path(resolve_path(str(raw), self.base_dir))
        if primary.exists():
            return str(primary)

        fallback = self.base_dir / "best.pt"
        return str(fallback if fallback.exists() else primary)

    def _load_detector(self, model_path: str):
        if YOLO is None:
            raise RuntimeError("ultralytics is required for label detection but is not installed.")
        if not Path(model_path).exists():
            raise RuntimeError(f"YOLO model not found at: {model_path}")
        return YOLO(model_path)

    def _resolve_model_id(self, model_id: str) -> str:
        if not model_id:
            return model_id
        path = Path(model_id)
        if not path.is_absolute():
            candidate = self.base_dir / path
            if candidate.exists():
                return str(candidate)
        return model_id

    def _load_ocr_model(self, model_id: str):
        image_processor = AutoImageProcessor.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = VisionEncoderDecoderModel.from_pretrained(model_id)
        return image_processor, tokenizer, model

    def _crop_foreground_pil(self, image: Image.Image) -> Image.Image:
        gray = ImageOps.grayscale(image)
        mask = gray.point(lambda p: 255 if p < 200 else 0, mode="1")
        bbox = mask.getbbox()
        if bbox:
            return image.crop(bbox)
        return image

    def _recognize_text_from_np(
        self,
        img_bgr: np.ndarray,
        center_crop_ratio: float,
        band_height_ratio: float,
        digits_only: bool = True,
    ) -> str:
        image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        if 0 < center_crop_ratio < 1.0:
            w, h = image.size
            crop_w, crop_h = int(w * center_crop_ratio), int(h * center_crop_ratio)
            left = (w - crop_w) // 2
            top = (h - crop_h) // 2
            image = image.crop((left, top, left + crop_w, top + crop_h))

        image = self._crop_foreground_pil(image)

        if 0 < band_height_ratio < 1.0:
            w, h = image.size
            band_h = int(h * band_height_ratio)
            top = (h - band_h) // 2
            image = image.crop((0, top, w, top + band_h))

        pixel_values = self.ocr_image_processor(images=image, return_tensors="pt").pixel_values.to(self.ocr_device)
        if self.ocr_device.type == "cuda" and next(self.ocr_model.parameters()).dtype == torch.float16:
            pixel_values = pixel_values.half()

        with torch.inference_mode():
            generated_ids = self.ocr_model.generate(
                pixel_values,
                num_beams=3,
                max_length=16,
                early_stopping=True,
            )

        generated_text = self.ocr_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        text = generated_text.strip()
        if digits_only:
            text = "".join(re.findall(r"\d", text))
        return text

    def _detect_best_box(self, frame: np.ndarray):
        if frame is None:
            return None, None
        results = self.detector(frame, conf=self.detector_conf)[0]
        boxes = getattr(results, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return results, None

        confs = getattr(boxes, "conf", None)
        best_idx = int(torch.argmax(confs).item()) if confs is not None and len(confs) else 0
        xyxy = boxes.xyxy[best_idx].tolist()
        x1, y1, x2, y2 = [int(round(v)) for v in xyxy]

        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(1, min(x2, w))
        y2 = max(1, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return results, None

        return results, (x1, y1, x2, y2)

    def process_frame(self, frame: np.ndarray):
        if frame is None:
            return False, "No frame available", None, None, None

        results, box = self._detect_best_box(frame)
        if box is None:
            annotated = results.plot() if results is not None else frame.copy()
            return False, "No paper detected", None, annotated, None

        x1, y1, x2, y2 = box
        cropped = frame[y1:y2, x1:x2].copy()
        if cropped.size == 0:
            annotated = results.plot() if results is not None else frame.copy()
            return False, "Invalid detected paper crop", None, annotated, None

        text = self._recognize_text_from_np(
            cropped,
            center_crop_ratio=self.ocr_center_crop_ratio,
            band_height_ratio=self.ocr_band_height_ratio,
            digits_only=self.ocr_digits_only,
        )

        annotated = results.plot() if results is not None else frame.copy()
        return True, "OK", text, annotated, cropped


def extract_target_digits(text: str, target_digits: int) -> str:
    digits = "".join(re.findall(r"\d", text or ""))
    if target_digits <= 0:
        return digits
    return digits[:target_digits] if len(digits) >= target_digits else ""
