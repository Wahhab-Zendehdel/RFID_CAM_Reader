import threading

from core.events import EventBuffer


class AppState:
    def __init__(self, rfid_cfg: dict, capture_cfg: dict, camera_cfg: dict, secondary_camera_cfg: dict):
        self.lock = threading.Lock()
        self.events = EventBuffer(maxlen=50)
        self.rfid_last_seen_by_tag = {}
        self.tag_submit_cooldowns = {}

        rfid_host = str((rfid_cfg or {}).get("host") or "").strip()
        rfid_port = int((rfid_cfg or {}).get("port") or 6000)

        self.rfid_status = {
            "enabled": bool(rfid_host),
            "host": rfid_host,
            "port": rfid_port,
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

        self.capture_status = {
            "state": "idle",
            "tag": "",
            "vehicle_id": None,
            "matched_tag": "",
            "started_time": None,
            "attempt": 0,
            "message": "",
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
            "warning_message": "",
            "awaiting_submit": False,
            "submitted_path": "",
            "submitted_time": None,
            "submitted_prefix": "",
            "primary_image_path": "",
            "secondary_image_path": "",
        }

        camera_source = str((camera_cfg or {}).get("rtsp_url") or "").strip()
        if not camera_source:
            if camera_cfg.get("index") is not None:
                camera_source = f"index:{camera_cfg.get('index')}"
            else:
                host = str((camera_cfg or {}).get("host") or "").strip()
                if host:
                    camera_source = host

        secondary_source = str((secondary_camera_cfg or {}).get("rtsp_url") or "").strip()
        if not secondary_source:
            if secondary_camera_cfg.get("index") is not None:
                secondary_source = f"index:{secondary_camera_cfg.get('index')}"
            else:
                host = str((secondary_camera_cfg or {}).get("host") or "").strip()
                if host:
                    secondary_source = host

        self.camera_status = {
            "enabled": bool(camera_source),
            "rtsp_url": camera_source,
            "connected": False,
            "reconnecting": False,
            "last_ok_ts": None,
            "last_error": "",
            "last_frame_time": None,
            "last_frame_ts": None,
            "consecutive_failures": 0,
        }

        self.secondary_camera_status = {
            "enabled": bool(secondary_source),
            "rtsp_url": secondary_source,
            "connected": False,
            "reconnecting": False,
            "last_ok_ts": None,
            "last_error": "",
            "last_frame_time": None,
            "last_frame_ts": None,
            "consecutive_failures": 0,
        }

        self.target_digits = int((capture_cfg or {}).get("target_digits") or 5)
        self.require_label_match_for_submit = bool(
            (capture_cfg or {}).get("require_label_match_for_submit", True)
        )
