import json
import logging

from src.utils.logger import JSONFormatter, get_logger, setup_logger, setup_logging


def test_json_formatter():
    fmt = JSONFormatter()
    record = logging.LogRecord("test", logging.INFO, "", 0, "hello world", (), None)
    output = fmt.format(record)
    parsed = json.loads(output)
    assert parsed["level"] == "INFO"
    assert parsed["message"] == "hello world"
    assert parsed["name"] == "test"
    assert "timestamp" in parsed


def test_setup_logger_console():
    logger = setup_logger("test-console", level=logging.DEBUG)
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) >= 1


def test_setup_logger_with_file(tmp_path):
    log_file = tmp_path / "test.log"
    logger = setup_logger("test-file", log_file=str(log_file), level=logging.INFO)
    logger.info("file log test")
    assert log_file.exists()
    content = log_file.read_text()
    assert "file log test" in content


def test_get_logger_default():
    logger = get_logger("test-get")
    assert logger is not None
    assert logger.name == "test-get"


def test_setup_logging_backward_compat():
    logger = setup_logging(verbose=True, name="test-backward")
    assert logger.level == logging.DEBUG


def test_setup_logging_default():
    logger = setup_logging(name="test-default-level")
    assert logger.level == logging.INFO


def test_json_file_format(tmp_path):
    log_file = tmp_path / "json.log"
    logger = setup_logger("json-test", log_file=str(log_file), json_format=True, level=logging.INFO)
    logger.info("json message")
    line = log_file.read_text().strip()
    parsed = json.loads(line)
    assert parsed["message"] == "json message"
    assert parsed["level"] == "INFO"
