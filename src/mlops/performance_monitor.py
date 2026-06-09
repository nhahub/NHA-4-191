import logging
import time
from collections import deque

import torch

logger = logging.getLogger(__name__)
logger.propagate = True


class PerformanceMonitor:
    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self.latencies: deque = deque(maxlen=window_size)
        self.request_count = 0
        self.error_count = 0
        self.start_time = time.monotonic()

    def record_request(self) -> None:
        self.request_count += 1

    def record_latency(self, latency_ms: float) -> None:
        self.latencies.append(latency_ms)

    def record_error(self) -> None:
        self.error_count += 1

    def get_stats(self) -> dict:
        uptime = time.monotonic() - self.start_time
        total = self.request_count
        errors = self.error_count
        error_rate = errors / total if total > 0 else 0.0

        stats = {
            "uptime_seconds": round(uptime, 2),
            "total_requests": total,
            "error_count": errors,
            "error_rate": round(error_rate, 4),
            "requests_per_second": round(total / uptime, 2) if uptime > 0 else 0.0,
        }

        if self.latencies:
            sorted_lats = sorted(self.latencies)
            n = len(sorted_lats)
            stats.update(
                {
                    "latency_ms_avg": round(sum(self.latencies) / n, 2),
                    "latency_ms_p50": round(sorted_lats[int(n * 0.50)], 2),
                    "latency_ms_p95": round(sorted_lats[int(n * 0.95)], 2),
                    "latency_ms_p99": round(sorted_lats[int(n * 0.99)], 2),
                }
            )

        if torch.cuda.is_available():
            stats["gpu_memory_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
            stats["gpu_memory_cached_mb"] = round(torch.cuda.memory_reserved() / (1024 * 1024), 2)

        return stats

    def log_stats(self) -> None:
        stats = self.get_stats()
        logger.info("Performance stats: %s", stats)
