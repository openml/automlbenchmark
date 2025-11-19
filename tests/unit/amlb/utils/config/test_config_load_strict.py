import os
import tempfile
import pytest
from amlb.utils.config import config_load
from amlb.utils import Namespace


def test_config_load_nonexistent_file_strict_false():
    """Non-existent file with strict=False should return empty Namespace."""
    result = config_load("/tmp/definitely_nonexistent_file_12345.yaml", strict=False)
    assert isinstance(result, Namespace)
    assert len(dir(result)) == 0  # Empty namespace


def test_config_load_nonexistent_file_strict_true():
    """Non-existent file with strict=True should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError) as exc_info:
        config_load("/tmp/definitely_nonexistent_file_12345.yaml", strict=True)

    error_message = str(exc_info.value)
    assert "not found" in error_message.lower()
    assert "typo" in error_message.lower()  # Should mention checking for typos


def test_config_load_existing_file_strict_false():
    """Existing file with strict=False should load normally."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("test_key: test_value\n")
        temp_file = f.name

    try:
        result = config_load(temp_file, strict=False)
        assert isinstance(result, Namespace)
        assert result.test_key == "test_value"
    finally:
        os.unlink(temp_file)


def test_config_load_existing_file_strict_true():
    """Existing file with strict=True should load normally."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("test_key: test_value\n")
        temp_file = f.name

    try:
        result = config_load(temp_file, strict=True)
        assert isinstance(result, Namespace)
        assert result.test_key == "test_value"
    finally:
        os.unlink(temp_file)


def test_config_load_default_is_strict_false():
    """Default behavior should be strict=False for backward compatibility."""
    result = config_load("/tmp/definitely_nonexistent_file_12345.yaml")
    assert isinstance(result, Namespace)
    assert len(dir(result)) == 0  # Empty namespace
