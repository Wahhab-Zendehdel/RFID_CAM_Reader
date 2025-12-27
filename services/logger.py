import json
import threading
import time
from pathlib import Path

from core.utils import utc_iso


class ProcessLogger:
    def __init__(self, state, logs_dir: str):
        self.state = state
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.logs_dir / "process.jsonl"
        self.file_lock = threading.Lock()

    def log(self, event_type: str, **fields) -> dict:
        now = time.time()
        payload = {"type": event_type, "time": now, "time_iso": utc_iso(now)}
        payload.update(fields)

        with self.file_lock:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")

        with self.state.lock:
            self.state.events.append(payload)

        return payload

    def recent_events(self) -> list:
        with self.state.lock:
            return self.state.events.list()
