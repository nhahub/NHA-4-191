import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)


class RoadSenseError(Exception):
    pass


class ConfigError(RoadSenseError):
    pass


class ModelLoadError(RoadSenseError):
    pass


class InferenceError(RoadSenseError):
    pass


class DataError(RoadSenseError):
    pass


class APIError(RoadSenseError):
    def __init__(self, message: str, status_code: int = 500, detail: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (IOError, ConnectionError, OSError),
) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            current_delay = delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        raise
                    logger.warning(
                        "%s failed (attempt %d/%d): %s. Retrying in %.1fs...",
                        func.__name__,
                        attempt,
                        max_attempts,
                        e,
                        current_delay,
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
            return None

        return wrapper

    return decorator
