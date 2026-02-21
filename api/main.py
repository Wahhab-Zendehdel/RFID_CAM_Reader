from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Query

from .schemas import (
    RecordListResponse,
    RecordOut,
    SortBy,
    SortDirection,
    StatsResponse,
)
from .store import RecordStore


def _get_store() -> RecordStore:
    config_path = os.environ.get("DB_CONFIG", "config/db.json")
    return RecordStore(config_path=config_path)


app = FastAPI(
    title="Capture Data API",
    description="Advanced CRUD API for capture data backed by SQLite with Swagger documentation.",
    version="1.3.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "capture-data-api", "version": "1.3.0"}


@app.get("/api/v1/records", response_model=RecordListResponse)
def list_records(
    station: Optional[str] = Query(default=None),
    station_id: Optional[int] = Query(default=None),
    station_type: Optional[str] = Query(default=None),
    tag: Optional[str] = Query(default=None, description="Filter by exact tag element in tags array"),
    number: Optional[str] = Query(default=None, description="Partial match on detected number"),
    has_errors: Optional[bool] = Query(default=None),
    search: Optional[str] = Query(default=None, description="Free-text search across station, number, tags, station_type"),
    date_from: Optional[str] = Query(default=None, description="ISO datetime lower bound"),
    date_to: Optional[str] = Query(default=None, description="ISO datetime upper bound"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    sort_by: SortBy = Query(default=SortBy.id),
    sort_dir: SortDirection = Query(default=SortDirection.desc),
) -> RecordListResponse:
    store = _get_store()
    rows, total = store.list_records(
        station=station,
        station_id=station_id,
        station_type=station_type,
        tag=tag,
        number=number,
        has_errors=has_errors,
        search=search,
        date_from=date_from,
        date_to=date_to,
        page=page,
        page_size=page_size,
        sort_by=sort_by.value,
        sort_dir=sort_dir.value,
    )
    return RecordListResponse(items=[RecordOut(**r) for r in rows], total=total, page=page, page_size=page_size)


@app.get("/api/v1/records/stats", response_model=StatsResponse)
def get_stats(
    station: Optional[str] = Query(default=None),
    station_id: Optional[int] = Query(default=None),
    date_from: Optional[str] = Query(default=None),
    date_to: Optional[str] = Query(default=None),
) -> StatsResponse:
    store = _get_store()
    data = store.get_stats(station=station, station_id=station_id, date_from=date_from, date_to=date_to)
    return StatsResponse(**data)


@app.get("/api/v1/records/{record_id}", response_model=RecordOut)
def get_record(record_id: int) -> RecordOut:
    store = _get_store()
    row = store.get_record(record_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Record not found")
    return RecordOut(**row)
