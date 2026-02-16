from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class RFIDStatus:
    enabled: bool = False
    host: str = ""
    port: int = 0
    connected: bool = False
    reconnecting: bool = False
    last_ok_ts: Optional[float] = None
    last_tag_ts: Optional[float] = None
    last_tag: str = ""
    last_tag_source: str = ""
    last_error: str = ""
    consecutive_failures: int = 0


@dataclass
class CameraStatus:
    name: str = ""
    enabled: bool = False
    source: str = ""
    connected: bool = False
    reconnecting: bool = False
    last_ok_ts: Optional[float] = None
    last_frame_ts: Optional[float] = None
    last_error: str = ""
    consecutive_failures: int = 0


@dataclass
class TagEvent:
    tag: str
    timestamp: float
    timestamp_iso: str
    source: str = ""
    raw_tag: str = ""
    success: bool = True
    error: str = ""
    rfid_connected: Optional[bool] = None
    rfid_error: str = ""


@dataclass
class CaptureResult:
    tag: str
    tag_source: str
    tag_timestamp: float
    tag_timestamp_iso: str
    started_ts: float
    finished_ts: Optional[float]
    success: bool
    number: str = ""
    raw_text: str = ""
    message: str = ""
    errors: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    primary_image: Optional[Any] = None
    secondary_image: Optional[Any] = None
    label_image: Optional[Any] = None
    annotated_image: Optional[Any] = None
    attempts: int = 0


@dataclass
class StationStatus:
    running: bool = False
    rfid: RFIDStatus = field(default_factory=RFIDStatus)
    primary_camera: Optional[CameraStatus] = None
    secondary_camera: Optional[CameraStatus] = None
    last_tag: Optional[TagEvent] = None
    last_capture: Optional[CaptureResult] = None
    last_error: str = ""
