"""Small, bounded process-local controls; shared deployments need shared admission."""
from __future__ import annotations

from collections import OrderedDict, defaultdict, deque
from copy import deepcopy
from threading import Lock
from time import monotonic


class CircuitOpen(RuntimeError):
    pass


class CircuitBreaker:
    def __init__(self, threshold=3, cooldown=20, clock=monotonic):
        self.threshold, self.cooldown, self.clock = threshold, cooldown, clock
        self.failures, self.opened, self.probing = 0, None, False
        self.lock = Lock()

    def call(self, operation):
        with self.lock:
            if self.opened is not None:
                if self.clock() - self.opened < self.cooldown or self.probing:
                    raise CircuitOpen("Provider temporarily unavailable; retry later")
                self.probing = True
        try:
            result = operation()
        except Exception:
            with self.lock:
                self.failures += 1
                if self.probing or self.failures >= self.threshold:
                    self.opened = self.clock()
                self.probing = False
            raise
        else:
            with self.lock:
                self.failures, self.opened, self.probing = 0, None, False
            return result


class TTLCache:
    def __init__(self, capacity=128, ttl=120, clock=monotonic):
        self.capacity, self.ttl, self.clock = capacity, ttl, clock
        self.entries, self.lock = OrderedDict(), Lock()

    def get(self, key):
        with self.lock:
            item = self.entries.get(key)
            if item is None:
                return None
            if item[0] <= self.clock():
                del self.entries[key]
                return None
            self.entries.move_to_end(key)
            return deepcopy(item[1])

    def put(self, key, value):
        with self.lock:
            self.entries[key] = (self.clock() + self.ttl, deepcopy(value))
            self.entries.move_to_end(key)
            while len(self.entries) > self.capacity:
                self.entries.popitem(last=False)


class RateLimiter:
    def __init__(self, limit=30, clock=monotonic):
        self.limit, self.clock = limit, clock
        self.entries, self.lock = defaultdict(deque), Lock()

    def admit(self, key):
        with self.lock:
            now = self.clock()
            for old in list(self.entries):
                if not self.entries[old] or self.entries[old][-1] <= now - 60:
                    del self.entries[old]
            bucket = self.entries[key]
            while bucket and bucket[0] <= now - 60:
                bucket.popleft()
            if len(bucket) >= self.limit:
                return False
            bucket.append(now)
            return True
