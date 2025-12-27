from collections import deque


class EventBuffer:
    def __init__(self, maxlen: int = 50):
        self._buffer = deque(maxlen=maxlen)

    def append(self, event: dict) -> None:
        self._buffer.append(event)

    def list(self) -> list:
        return list(self._buffer)
