import re
from datetime import datetime, timezone
from typing import Tuple


def normalize_tag(tag: str) -> str:
    tag = (tag or "").strip().upper()
    return re.sub(r"[^0-9A-F]", "", tag)


def normalize_label_value(label) -> str:
    if label is None:
        return ""
    label_text = str(label).strip()
    if not label_text:
        return ""
    return label_text.upper()


def validate_label_input(label: str, target_digits: int) -> Tuple[bool, str, str]:
    if label is None:
        return True, "", ""
    label_text = str(label).strip()
    if not label_text:
        return True, "", ""
    label_norm = label_text.upper()
    if target_digits > 0 and label_norm.isdigit() and len(label_norm) != target_digits:
        return False, "", f"Label must be {target_digits} digits"
    return True, label_norm, ""


def utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def safe_component(text: str, fallback: str = "unknown") -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_-]+", "_", (text or "").strip())
    return cleaned or fallback
