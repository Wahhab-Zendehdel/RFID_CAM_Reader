from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
from transformers import logging as _tf_logging
try:
    # TrOCR has a dedicated processor wrapper
    from transformers import TrOCRProcessor
except Exception:  # pragma: no cover - optional presence
    TrOCRProcessor = None

from .common_config import resolve_path


class LabelDetector:
    def __init__(self, paper_cfg: dict, ocr_cfg: dict, base_dir: Optional[Path] = None) -> None:
        self.paper_cfg = paper_cfg or {}
        self.ocr_cfg = ocr_cfg or {}
        self.base_dir = base_dir or Path.cwd()

        self.mask_path = resolve_path(self.paper_cfg.get("mask_path") or "Mask.png", self.base_dir)
        self.min_area = int(self.paper_cfg.get("min_area") or 3000)
        self.max_area_frac = float(self.paper_cfg.get("max_area_frac") or 0.10)
        self.match_width = int(self.paper_cfg.get("match_width") or 300)
        self.min_tm_score = float(self.paper_cfg.get("min_tm_score") or 1e-12)
        self.color_std_min = float(self.paper_cfg.get("color_std_min") or 40.0)

        self.ocr_model_id = str(self.ocr_cfg.get("model_id") or "./trocr-large-printed")
        self.ocr_center_crop_ratio = float(self.ocr_cfg.get("center_crop_ratio") or 0.8)
        self.ocr_band_height_ratio = float(self.ocr_cfg.get("band_height_ratio") or 0.5)
        self.ocr_digits_only = bool(self.ocr_cfg.get("digits_only", True))

        self.mask_resized, self.mask_aspect = self._load_mask()

        model_id = self._resolve_model_id(self.ocr_model_id)
        self.ocr_image_processor, self.ocr_tokenizer, self.ocr_model = self._load_ocr_model(model_id)
        # Allow overriding device via OCR config (e.g. "cpu" or "cuda").
        requested = str(self.ocr_cfg.get("device") or "").strip().lower()
        if requested in ("cpu", "cuda"):
            if requested == "cuda" and not torch.cuda.is_available():
                # fallback to cpu if cuda requested but unavailable
                self.ocr_device = torch.device("cpu")
            else:
                self.ocr_device = torch.device(requested)
        else:
            self.ocr_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ocr_model.to(self.ocr_device)
        self.ocr_model.eval()
        # Only use half precision when running on CUDA and model parameters are float16-capable.
        if self.ocr_device.type == "cuda":
            try:
                self.ocr_model.half()
            except Exception:
                pass

    def _resolve_model_id(self, model_id: str) -> str:
        if not model_id:
            return model_id
        path = Path(model_id)
        if not path.is_absolute():
            candidate = self.base_dir / path
            if candidate.exists():
                return str(candidate)
        return model_id

    def _load_mask(self):
        mask_img = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            # If no mask file is present, fall back to a reasonable default mask
            # to allow the detector to be instantiated in demo/test environments.
            # mh, mw = 100, self.match_width
            # mask_img = 255 * (np.ones((mh, mw), dtype=np.uint8))
            raise IOError("Mask not found at: " + self.mask_path)
        mh, mw = mask_img.shape[:2]
        scale = self.match_width / float(mw)
        mask_resized = cv2.resize(mask_img, (self.match_width, int(mh * scale)))
        mh, mw = mask_resized.shape[:2]
        mask_aspect = mw / float(mh)
        # Optionally dump mask for debugging
        # try:
        #     import os
        #     if os.environ.get("PAPER_DETECT_DEBUG") == "1":
        #         dbg_dir = Path(self.base_dir) / "debug_paper"
        #         dbg_dir.mkdir(parents=True, exist_ok=True)
        #         cv2.imwrite(str(dbg_dir / "mask_resized.png"), mask_resized)
        # except Exception:
        #     pass
        return mask_resized, mask_aspect

    def _load_ocr_model(self, model_id: str):
        # Reduce transformers verbosity for the model load report
        # try:
        #     _tf_logging.set_verbosity_error()
        # except Exception:
        #     pass

        # # Prefer TrOCRProcessor when available for microsoft/trocr models
        # if TrOCRProcessor is not None:
        #     try:
        #         proc = TrOCRProcessor.from_pretrained(model_id)
        #         # processor may expose either image_processor or feature_extractor
        #         image_processor = getattr(proc, "image_processor", None) or getattr(proc, "feature_extractor", None)
        #         tokenizer = getattr(proc, "tokenizer", None)
        #         # Try to fully materialize model weights (avoid meta/device placeholder tensors)
        #         try:
        #             model = VisionEncoderDecoderModel.from_pretrained(model_id, low_cpu_mem_usage=False)
        #         except TypeError:
        #             model = VisionEncoderDecoderModel.from_pretrained(model_id)
        #         if image_processor is not None and tokenizer is not None:
        #             return image_processor, tokenizer, model
        #     except Exception:
        #         # fallback to generic loaders
        #         pass

        image_processor = AutoImageProcessor.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # If the model is stored as safetensors or was saved with memory optimizations,
        # force full materialization to avoid tensors on the 'meta' device which later
        # cause errors during generation.
        # try:
        #     model = VisionEncoderDecoderModel.from_pretrained(model_id, low_cpu_mem_usage=False)
        # except TypeError:
        #     model = VisionEncoderDecoderModel.from_pretrained(model_id)
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
        # Keep input dtype compatible with model/device: only half on CUDA when model uses float16
        # if self.ocr_device.type == "cuda":
        #     try:
        #         if next(self.ocr_model.parameters()).dtype == torch.float16:
        #             pixel_values = pixel_values.half()
        #     except Exception:
        #         pass

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

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _is_rectangle_by_angles(self, pts: np.ndarray, min_angle=70.0, max_angle=110.0) -> bool:
        def angle(a, b, c):
            ba = a - b
            bc = c - b
            cos_angle = np.dot(ba, bc) / ((np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8)
            return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

        for i in range(4):
            a = pts[i]
            b = pts[(i + 1) % 4]
            c = pts[(i + 2) % 4]
            ang = angle(a, b, c)
            if ang < min_angle or ang > max_angle:
                return False
        return True

    def find_best_candidate_single_frame(self, frame: np.ndarray):
        if frame is None:
            return None, None, -1.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 30, 100)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_h * frame_w

        mh, mw = self.mask_resized.shape[:2]
        dest_quad = np.array([[0, 0], [mw, 0], [mw, mh], [0, mh]], dtype=np.float32)

        best_score = -1.0
        best_quad = None
        best_warped = None
        # considered = 0

        # Track largest rectangle candidate as a fallback when template matching fails
        # largest_area = 0.0
        # largest_quad = None
        # largest_warped = None
        # largest_std = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            # considered += 1
            if area > self.max_area_frac * frame_area:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) != 4:
                continue

            pts = self._order_points(approx.reshape(4, 2))

            if not self._is_rectangle_by_angles(pts, min_angle=70.0, max_angle=110.0):
                continue

            w_edge = np.linalg.norm(pts[1] - pts[0])
            h_edge = np.linalg.norm(pts[3] - pts[0])
            if h_edge == 0:
                continue
            aspect = w_edge / h_edge

            if not (0.5 * self.mask_aspect <= aspect <= 2.0 * self.mask_aspect):
                continue

            M = cv2.getPerspectiveTransform(pts.astype(np.float32), dest_quad)
            warped = cv2.warpPerspective(frame, M, (mw, mh))
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

            mean_val, std_val = cv2.meanStdDev(warped_gray)
            std_intensity = float(std_val[0][0])
            if std_intensity < self.color_std_min:
                continue

            # remember largest rectangle in case template matching doesn't find a good match
            # if area > largest_area:
            #     largest_area = area
            #     largest_quad = pts
            #     largest_warped = warped
            #     largest_std = std_intensity

            result = cv2.matchTemplate(warped_gray, self.mask_resized, cv2.TM_CCOEFF_NORMED)
            _, score, _, _ = cv2.minMaxLoc(result)

            if score > best_score:
                best_score = score
                best_quad = pts
                best_warped = warped
            # Debug: save candidate warps and match maps when enabled
            # try:
            #     import os
            #     if os.environ.get("PAPER_DETECT_DEBUG") == "1":
            #         dbg_dir = Path(self.base_dir) / "debug_paper"
            #         dbg_dir.mkdir(parents=True, exist_ok=True)
            #         idx = considered
            #         # save warped color and a visualized match map
            #         try:
            #             cv2.imwrite(str(dbg_dir / f"candidate_{idx:03d}_score_{score:.3f}.png"), warped)
            #         except Exception:
            #             pass
            #         try:
            #             # normalize result to 0-255 for visualization
            #             rmin, rmax = result.min(), result.max()
            #             if rmax - rmin > 1e-6:
            #                 vis = ((result - rmin) / (rmax - rmin) * 255.0).astype('uint8')
            #             else:
            #                 vis = (result * 0).astype('uint8')
            #             cv2.imwrite(str(dbg_dir / f"candidate_{idx:03d}_matchmap_{score:.3f}.png"), vis)
            #         except Exception:
            #             pass
            # except Exception:
            #     pass

        # If template matching didn't find anything, fall back to the largest candidate
        # if best_quad is None and largest_quad is not None:
        #     if largest_std >= self.color_std_min:
        #         best_quad = largest_quad
        #         best_warped = largest_warped
        #         best_score = 0.0

        if best_quad is None or best_score < self.min_tm_score:
            # Optional debug: print a short diagnostic when enabled via env var
            # try:
            #     import os

            #     if os.environ.get("PAPER_DETECT_DEBUG") == "1":
            #         print(f"Paper detection: considered_contours={considered} total_contours={len(contours)} frame_area={frame_area} best_score={best_score} min_area={self.min_area} max_area_frac={self.max_area_frac} color_std_min={self.color_std_min}")
            # except Exception:
            #     pass
            return None, None, -1.0

        return best_quad, best_warped, best_score

    def process_frame(self, frame: np.ndarray):
        if frame is None:
            return False, "No frame available", None, None, None

        quad, warped, score = self.find_best_candidate_single_frame(frame)
        if quad is None:
            annotated = frame.copy()
            return False, "No paper detected", None, annotated, None

        text = self._recognize_text_from_np(
            warped,
            center_crop_ratio=self.ocr_center_crop_ratio,
            band_height_ratio=self.ocr_band_height_ratio,
            digits_only=self.ocr_digits_only,
        )

        annotated = frame.copy()
        cv2.drawContours(annotated, [quad.astype(int)], -1, (0, 255, 0), 3)

        return True, "OK", text, annotated, warped


def extract_target_digits(text: str, target_digits: int) -> str:
    digits = "".join(re.findall(r"\d", text or ""))
    if target_digits <= 0:
        return digits
    return digits[:target_digits] if len(digits) >= target_digits else ""
