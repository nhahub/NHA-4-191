import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


_LOG_FILE: Path | None = None


def setup_logger(
    name: str = "road_sense",
    log_file: str | None = None,
    level: int = logging.INFO,
    json_format: bool = False,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    if json_format:
        console_handler.setFormatter(JSONFormatter(datefmt="%Y-%m-%dT%H:%M:%S"))
    else:
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    logger.addHandler(console_handler)

    if log_file:
        global _LOG_FILE
        _LOG_FILE = Path(log_file)
        _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(str(_LOG_FILE), maxBytes=max_bytes, backupCount=backup_count)
        file_handler.setFormatter(JSONFormatter(datefmt="%Y-%m-%dT%H:%M:%S"))
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "road_sense") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


def setup_logging(verbose: bool = False, name: str = "road_sense", log_file: str | None = None) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    return setup_logger(name=name, level=level, log_file=log_file)
