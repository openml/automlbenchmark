import pytest
from amlb.benchmarks.file import load_file_benchmark


def test_missing_benchmark_file_raises_error():
    """Missing benchmark file should raise ValueError when finding it, then FileNotFoundError when loading it."""
    with pytest.raises(ValueError) as exc_info:
        load_file_benchmark("nonexistent_benchmark", ["/tmp/nonexistent_dir"])

    error_message = str(exc_info.value)
    assert "incorrect benchmark" in error_message.lower()


def test_benchmark_file_with_typo_in_path_raises_error():
    """If a full path to benchmark is provided but has a typo, it should raise ValueError."""
    with pytest.raises(ValueError) as exc_info:
        load_file_benchmark(
            "/tmp/nonexistent_benchmark_with_typo.yaml",
            ["/tmp"],  # directory exists, but file doesn't
        )

    error_message = str(exc_info.value)
    assert "incorrect benchmark" in error_message.lower()
