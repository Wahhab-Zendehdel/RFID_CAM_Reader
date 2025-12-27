import base64
import json
import queue
import re
import threading
import time
from datetime import datetime
from typing import Optional, Tuple
from pathlib import Path

import cv2

from core.config import resolve_path
from core.utils import safe_component, utc_iso


class Orchestrator:
    def __init__(self, cfg: dict, state, repository, camera_service, label_detector, validator, logger):
        self.state = state
        self.repository = repository
        self.camera_service = camera_service
        self.label_detector = label_detector
        self.validator = validator
        self.logger = logger

        capture_cfg = cfg.get("capture", {})
        timing_cfg = cfg.get("timing", {})
        rfid_cfg = cfg.get("rfid", {})
        storage_cfg = cfg.get("storage", {})

        self.retry_interval_seconds = float(capture_cfg.get("retry_interval_seconds") or 0.4)
        self.timeout_seconds = float(capture_cfg.get("timeout_seconds") or 30.0)
        self.target_digits = int(capture_cfg.get("target_digits") or 5)
        self.require_label_match_for_submit = bool(capture_cfg.get("require_label_match_for_submit", True))

        self.tag_cooldown_seconds = float(timing_cfg.get("tag_cooldown_seconds") or 3.0)
        self.tag_submit_cooldown_seconds = float(timing_cfg.get("tag_submit_cooldown_seconds") or 300.0)

        self.present_window_seconds = float(rfid_cfg.get("present_window_seconds") or 2.0)
        self.queue_maxsize = int(rfid_cfg.get("queue_maxsize") or 50)

        self.submissions_dir = Path(resolve_path("submissions"))
        self.submissions_dir.mkdir(parents=True, exist_ok=True)

        images_dir = storage_cfg.get("images_dir") or "data/images"
        self.images_dir = Path(resolve_path(images_dir))
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.queue = queue.Queue(maxsize=self.queue_maxsize)
        self.running = False
        self.thread = None

    def start(self) -> None:
        if self.thread is not None:
            return
        self.running = True
        self.thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)

    def enqueue_tag(self, tag: str, source: str) -> None:
        if not tag:
            return

        now = time.time()
        with self.state.lock:
            last_num_time = self.state.capture_status.get("number_time") or self.state.capture_status.get("started_time") or 0
            last_tag_active = self.state.capture_status.get("tag") == tag and self.state.capture_status.get("state") in {
                "running",
                "success",
                "awaiting_submit",
                "submitted",
            }
            pending_submit = bool(self.state.capture_status.get("awaiting_submit"))
            cooldown_until = float(self.state.tag_submit_cooldowns.get(tag) or 0.0)

        if pending_submit:
            self.logger.log("blocked", tag=tag, reason="awaiting_submit")
            return

        if cooldown_until and now < cooldown_until:
            self.logger.log(
                "cooldown",
                tag=tag,
                cooldown_until=cooldown_until,
                cooldown_until_iso=utc_iso(cooldown_until),
            )
            return

        if last_tag_active and self.tag_cooldown_seconds > 0 and (now - last_num_time) < self.tag_cooldown_seconds:
            return

        with self.state.lock:
            current_tag = self.state.capture_status.get("tag")
            current_state = self.state.capture_status.get("state")

        if current_state == "running" and current_tag == tag:
            return

        try:
            queued_tags = list(self.queue.queue)
        except Exception:
            queued_tags = []
        if tag in queued_tags:
            return

        self._purge_tag_from_queue(tag)

        try:
            self.queue.put_nowait(tag)
            self.logger.log("queue", tag=tag, source=source)
        except queue.Full:
            with self.state.lock:
                self.state.rfid_status["last_error"] = "RFID queue full"

    def _purge_tag_from_queue(self, tag: str) -> None:
        try:
            items = []
            while True:
                items.append(self.queue.get_nowait())
        except queue.Empty:
            pass
        else:
            for item in items:
                if item != tag:
                    try:
                        self.queue.put_nowait(item)
                    except queue.Full:
                        break

    def _clear_queue(self) -> int:
        cleared = 0
        try:
            while True:
                self.queue.get_nowait()
                cleared += 1
        except queue.Empty:
            pass
        return cleared

    def _capture_worker(self) -> None:
        while self.running:
            try:
                tag = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                self._process_capture_for_tag(tag)
            finally:
                try:
                    self.queue.task_done()
                except Exception:
                    pass

    def _process_capture_for_tag(self, tag: str) -> None:
        start = time.time()

        vehicle = self.validator.resolve_vehicle(tag)
        if not vehicle:
            with self.state.lock:
                self.state.capture_status.update(
                    {
                        "state": "failed",
                        "tag": tag,
                        "vehicle_id": None,
                        "matched_tag": "",
                        "started_time": start,
                        "attempt": 0,
                        "message": "Unknown tag",
                        "raw_text": "",
                        "number": "",
                        "number_time": None,
                        "original_b64": None,
                        "paper_b64": None,
                        "secondary_b64": None,
                        "secondary_message": "",
                        "label_expected": "",
                        "expected_labels": [],
                        "label_detected": "",
                        "label_match": None,
                        "label_message": "",
                        "warning_message": "Unknown tag",
                        "awaiting_submit": False,
                        "submitted_path": "",
                        "submitted_time": None,
                        "submitted_prefix": "",
                        "primary_image_path": "",
                        "secondary_image_path": "",
                    }
                )
            self.logger.log("unknown_tag", tag=tag)
            return

        vehicle_id = vehicle.get("id")
        expected_labels = [label for label in (vehicle.get("labels") or []) if label]
        self.logger.log("vehicle_resolved", tag=tag, vehicle_id=vehicle_id)

        with self.state.lock:
            self.state.capture_status.update(
                {
                    "state": "running",
                    "tag": tag,
                    "vehicle_id": vehicle_id,
                    "matched_tag": tag,
                    "started_time": start,
                    "attempt": 0,
                    "message": "Capturing...",
                    "raw_text": "",
                    "number": "",
                    "number_time": None,
                    "original_b64": None,
                    "paper_b64": None,
                    "secondary_b64": None,
                    "secondary_message": "",
                    "label_expected": expected_labels[0] if expected_labels else "",
                    "expected_labels": expected_labels,
                    "label_detected": "",
                    "label_match": None,
                    "label_message": "",
                    "warning_message": "",
                    "awaiting_submit": False,
                    "submitted_path": "",
                    "submitted_time": None,
                    "submitted_prefix": "",
                    "primary_image_path": "",
                    "secondary_image_path": "",
                }
            )

        self.logger.log("capture_started", tag=tag, vehicle_id=vehicle_id)

        attempt = 0
        while self.running:
            now = time.time()
            if (now - start) > self.timeout_seconds:
                with self.state.lock:
                    self.state.capture_status.update(
                        {
                            "state": "failed",
                            "message": f"Timed out after {attempt} attempts",
                            "number_time": now,
                            "awaiting_submit": False,
                            "label_expected": expected_labels[0] if expected_labels else "",
                            "expected_labels": expected_labels,
                            "label_detected": "",
                            "label_match": None,
                            "label_message": "",
                            "warning_message": "Timed out",
                        }
                    )
                self.logger.log("timeout", tag=tag, vehicle_id=vehicle_id, attempt=attempt)
                return

            attempt += 1
            frame = self.camera_service.get_primary_frame()
            success, message, text, annotated, warped = self.label_detector.process_frame(frame)
            digits = self._extract_target_digits(text) if success else ""

            original_b64 = None
            paper_b64 = None
            try:
                if annotated is not None:
                    original_b64 = self._img_to_base64_jpeg(annotated)
            except Exception:
                original_b64 = None

            if warped is not None:
                try:
                    paper_b64 = self._img_to_base64_jpeg(warped)
                except Exception:
                    paper_b64 = None

            with self.state.lock:
                self.state.capture_status["attempt"] = attempt
                self.state.capture_status["message"] = message
                self.state.capture_status["raw_text"] = text or ""
                self.state.capture_status["original_b64"] = original_b64
                self.state.capture_status["paper_b64"] = paper_b64

            self.logger.log("ocr_attempt", tag=tag, vehicle_id=vehicle_id, attempt=attempt, message=message)

            if digits:
                done = time.time()
                sec_b64, sec_msg = self._grab_secondary_snapshot()
                validation = self.validator.validate_label(vehicle, digits, tag)

                self.logger.log(
                    "number",
                    tag=tag,
                    vehicle_id=vehicle_id,
                    number=digits,
                    attempt=attempt,
                )

                label_match = validation.get("label_match")
                warning_message = validation.get("message")

                with self.state.lock:
                    self.state.capture_status.update(
                        {
                            "state": "awaiting_submit",
                            "number": digits,
                            "number_time": done,
                            "message": "Captured. Awaiting submit.",
                            "awaiting_submit": True,
                            "submitted_path": "",
                            "submitted_time": None,
                            "submitted_prefix": "",
                            "secondary_b64": sec_b64,
                            "secondary_message": sec_msg or "",
                            "label_expected": (validation.get("expected_labels") or [""])[0],
                            "expected_labels": validation.get("expected_labels", []),
                            "label_detected": validation.get("detected_label", ""),
                            "label_match": label_match,
                            "label_message": warning_message,
                            "warning_message": warning_message,
                            "matched_tag": validation.get("matched_tag") or tag,
                        }
                    )

                self.logger.log(
                    "ocr_success",
                    tag=tag,
                    vehicle_id=vehicle_id,
                    detected_label=validation.get("detected_label", ""),
                )
                if label_match is False:
                    self.logger.log(
                        "mismatch",
                        tag=tag,
                        vehicle_id=vehicle_id,
                        expected=",".join(validation.get("expected_labels") or []),
                        detected=validation.get("detected_label", ""),
                    )
                else:
                    self.logger.log(
                        "validation",
                        tag=tag,
                        vehicle_id=vehicle_id,
                        match=label_match,
                    )

                primary_path, secondary_path = self._store_vehicle_images(
                    vehicle_id, annotated, self.camera_service.get_secondary_frame()
                )
                with self.state.lock:
                    self.state.capture_status["primary_image_path"] = primary_path or ""
                    self.state.capture_status["secondary_image_path"] = secondary_path or ""

                self.repository.upsert_vehicle_images(vehicle_id, primary_path, secondary_path)

                self._clear_queue()
                return

            time.sleep(self.retry_interval_seconds)

    def submit_capture(self) -> Tuple[bool, str, Optional[dict]]:
        with self.state.lock:
            capture = dict(self.state.capture_status)

        if not capture.get("awaiting_submit"):
            return False, "No capture awaiting submit", None
        if not capture.get("number"):
            return False, "No number captured", None
        if self.require_label_match_for_submit and capture.get("label_match") is False:
            message = capture.get("warning_message") or capture.get("label_message") or "Label mismatch; submit blocked"
            return False, message, None

        effective_tag = capture.get("matched_tag") or capture.get("tag") or ""
        save_capture = dict(capture)
        if capture.get("tag") and capture.get("tag") != effective_tag:
            save_capture["tag_original"] = capture.get("tag")
        save_capture["tag"] = effective_tag

        saved = self._save_capture_to_disk(save_capture)

        now = time.time()
        with self.state.lock:
            self.state.capture_status["awaiting_submit"] = False
            self.state.capture_status["state"] = "submitted"
            self.state.capture_status["message"] = "Submitted"
            self.state.capture_status["submitted_path"] = saved["files"].get("json", "")
            self.state.capture_status["submitted_time"] = saved.get("timestamp")
            self.state.capture_status["submitted_prefix"] = saved.get("prefix", "")
            if effective_tag:
                self.state.tag_submit_cooldowns[effective_tag] = now + self.tag_submit_cooldown_seconds

        self.logger.log(
            "submitted",
            tag=effective_tag,
            tag_original=capture.get("tag") if effective_tag and capture.get("tag") != effective_tag else "",
            number=capture.get("number") or "",
            label_detected=capture.get("label_detected") or "",
            label_match=capture.get("label_match"),
            path=saved["files"].get("json", ""),
        )

        return True, "Submitted", saved

    def _store_vehicle_images(self, vehicle_id: int, primary_frame, secondary_frame):
        ts_label = datetime.utcfromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
        primary_path = None
        secondary_path = None

        if primary_frame is not None:
            primary_path = str(self.images_dir / f"vehicle_{vehicle_id}_{ts_label}_primary.jpg")
            try:
                cv2.imwrite(primary_path, primary_frame)
            except Exception:
                primary_path = None

        if secondary_frame is not None:
            secondary_path = str(self.images_dir / f"vehicle_{vehicle_id}_{ts_label}_secondary.jpg")
            try:
                cv2.imwrite(secondary_path, secondary_frame)
            except Exception:
                secondary_path = None

        return primary_path, secondary_path

    def _extract_target_digits(self, text: str) -> str:
        digits = "".join(re.findall(r"\d", text or ""))
        if self.target_digits <= 0:
            return digits
        return digits[: self.target_digits] if len(digits) >= self.target_digits else ""

    def _img_to_base64_jpeg(self, img_bgr) -> str:
        ok, buf = cv2.imencode(".jpg", img_bgr)
        if not ok:
            raise RuntimeError("Failed to encode image")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def _grab_secondary_snapshot(self):
        frame = self.camera_service.get_secondary_frame()
        if frame is None:
            return None, "Secondary frame not available"
        try:
            return self._img_to_base64_jpeg(frame), "OK"
        except Exception as exc:
            return None, f"Secondary encode failed: {exc}"

    def _save_capture_to_disk(self, capture: dict) -> dict:
        ts = time.time()
        ts_label = datetime.utcfromtimestamp(ts).strftime("%Y%m%d_%H%M%S")

        tag = safe_component(capture.get("tag") or "unknown", "tag")
        number = safe_component(capture.get("number") or "-----", "number")
        attempt = int(capture.get("attempt") or 0)

        prefix = f"{ts_label}_TAG-{tag}_NUM-{number}_ATT-{attempt:02d}"
        json_path = self.submissions_dir / f"{prefix}.json"
        orig_path = self.submissions_dir / f"{prefix}_original.jpg"
        paper_path = self.submissions_dir / f"{prefix}_paper.jpg"
        secondary_path = self.submissions_dir / f"{prefix}_secondary.jpg"

        meta = {
            "tag": capture.get("tag") or "",
            "number": capture.get("number") or "",
            "raw_text": capture.get("raw_text") or "",
            "attempt": attempt,
            "message": capture.get("message") or "",
            "secondary_message": capture.get("secondary_message") or "",
            "timestamp": ts,
            "timestamp_iso": utc_iso(ts),
            "files": {},
        }
        if capture.get("tag_original"):
            meta["tag_original"] = capture.get("tag_original")
        if capture.get("matched_tag"):
            meta["matched_tag"] = capture.get("matched_tag")
        if capture.get("vehicle_id"):
            meta["vehicle_id"] = capture.get("vehicle_id")

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
