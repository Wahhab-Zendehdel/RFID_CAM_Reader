from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class SortBy(str, Enum):
    id = "id"
    datetime = "datetime"
    created_at = "created_at"
    station = "station"
    number = "number"
    station_type = "station_type"


class SortDirection(str, Enum):
    asc = "asc"
    desc = "desc"


class RecordCreate(BaseModel):
    datetime: Optional[str] = None
    station: str = Field(..., min_length=1)
    station_id: Optional[int] = None
    station_type: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    number: Optional[str] = None
    primary_image_url: Optional[str] = None
    secondary_image_url: Optional[str] = None
    label_image_url: Optional[str] = None
    rfid_device_id: Optional[int] = None
    primary_cam_id: Optional[int] = None
    secondary_cam_id: Optional[int] = None
    errors: Optional[Any] = None


class RecordUpdate(BaseModel):
    datetime: Optional[str] = None
    station: Optional[str] = Field(default=None, min_length=1)
    station_id: Optional[int] = None
    station_type: Optional[str] = None
    tags: Optional[List[str]] = None
    number: Optional[str] = None
    primary_image_url: Optional[str] = None
    secondary_image_url: Optional[str] = None
    label_image_url: Optional[str] = None
    rfid_device_id: Optional[int] = None
    primary_cam_id: Optional[int] = None
    secondary_cam_id: Optional[int] = None
    errors: Optional[Any] = None
    created_at: Optional[str] = None
    synced: Optional[int] = None
    mysql_id: Optional[int] = None
    last_error: Optional[str] = None


class RecordOut(BaseModel):
    id: int
    datetime: Optional[str] = None
    station: Optional[str] = None
    station_id: Optional[int] = None
    station_type: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    number: Optional[str] = None
    primary_image_url: Optional[str] = None
    secondary_image_url: Optional[str] = None
    label_image_url: Optional[str] = None
    rfid_device_id: Optional[int] = None
    primary_cam_id: Optional[int] = None
    secondary_cam_id: Optional[int] = None
    errors: Optional[Any] = None
    created_at: Optional[str] = None
    synced: Optional[int] = None
    mysql_id: Optional[int] = None
    last_error: Optional[str] = None


class RecordListResponse(BaseModel):
    items: List[RecordOut]
    total: int
    page: int
    page_size: int


class BulkCreateRequest(BaseModel):
    records: List[RecordCreate] = Field(..., min_items=1, max_items=500)


class BulkCreateResponse(BaseModel):
    inserted: int
    ids: List[int]


class StationCount(BaseModel):
    station: Optional[str] = None
    count: int


class StatsResponse(BaseModel):
    total_records: int
    records_with_errors: int
    records_with_number: int
    date_min: Optional[str] = None
    date_max: Optional[str] = None
    by_station: List[StationCount]
    by_station_type: List[StationCount]

