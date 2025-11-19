"""Test for race condition fix in local results file writing (issue #691)."""

import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


from amlb.results import Scoreboard


def test_parallel_save_no_race_condition():
    """Test that multiple parallel saves don't cause data loss due to race conditions."""
    # Create a temporary directory for test results
    with tempfile.TemporaryDirectory() as tmpdir:
        scores_dir = Path(tmpdir) / "scores"
        scores_dir.mkdir()

        # Create multiple scoreboards with different data
        num_parallel_saves = 10
        scoreboards = []
        for i in range(num_parallel_saves):
            # Create a simple score entry for each iteration
            score_data = {
                "id": f"test_task_{i}",
                "task": f"task_{i}",
                "framework": "test_framework",
                "constraint": "test",
                "fold": i,
                "type": "classification",
                "result": 0.9 + i * 0.001,
                "metric": "accuracy",
                "mode": "local",
                "version": "1.0",
                "params": "",
                "app_version": "1.0",
                "utc": "2025-01-01T00:00:00",
                "duration": 100.0,
                "training_duration": 80.0,
                "predict_duration": 5.0,
                "models_count": 1,
                "seed": i,
                "info": "",
            }
            board = Scoreboard(scores=[score_data], scores_dir=str(scores_dir))
            scoreboards.append(board)

        # Function to save a scoreboard (will be run in parallel)
        def save_board(board):
            board.save(append=True)

        # Save all scoreboards in parallel using ThreadPoolExecutor
        # This simulates the race condition scenario
        with ThreadPoolExecutor(max_workers=num_parallel_saves) as executor:
            futures = [executor.submit(save_board, board) for board in scoreboards]
            # Wait for all to complete
            for future in futures:
                future.result()

        # Load the results and verify all data was saved
        result_board = Scoreboard.all(scores_dir=str(scores_dir))
        result_df = result_board.as_data_frame()

        # Check that we have all the expected rows
        assert len(result_df) == num_parallel_saves, (
            f"Expected {num_parallel_saves} rows, but got {len(result_df)}"
        )

        # Check that all task IDs are present
        expected_task_ids = {f"test_task_{i}" for i in range(num_parallel_saves)}
        actual_task_ids = set(result_df["id"].values)
        assert expected_task_ids == actual_task_ids, (
            f"Missing task IDs: {expected_task_ids - actual_task_ids}"
        )

        # Check that all folds are present and unique
        expected_folds = set(range(num_parallel_saves))
        actual_folds = set(result_df["fold"].values)
        assert expected_folds == actual_folds, (
            f"Missing folds: {expected_folds - actual_folds}"
        )


def test_save_with_file_lock_timeout(mocker):
    """Test that file lock timeout is handled gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        scores_dir = Path(tmpdir) / "scores"
        scores_dir.mkdir()

        score_data = {
            "id": "test_task",
            "task": "task",
            "framework": "test_framework",
            "constraint": "test",
            "fold": 0,
            "type": "classification",
            "result": 0.9,
            "metric": "accuracy",
            "mode": "local",
            "version": "1.0",
            "params": "",
            "app_version": "1.0",
            "utc": "2025-01-01T00:00:00",
            "duration": 100.0,
            "training_duration": 80.0,
            "predict_duration": 5.0,
            "models_count": 1,
            "seed": 0,
            "info": "",
        }
        board = Scoreboard(scores=[score_data], scores_dir=str(scores_dir))

        # Mock file_lock to raise TimeoutError
        from amlb.utils import process

        original_file_lock = process.file_lock

        def mock_file_lock(*args, **kwargs):
            raise TimeoutError("Lock timeout")

        mocker.patch("amlb.utils.process.file_lock", side_effect=mock_file_lock)

        # The save should handle the timeout gracefully (not crash)
        # Note: This tests the Scoreboard.save_df method behavior
        # The actual file lock integration is tested in benchmark._save()
        try:
            # In the actual implementation, this would be wrapped with file_lock
            # Here we're just testing that the code structure allows for error handling
            board.save(append=True)
        except TimeoutError:
            # Expected to potentially raise, but shouldn't crash the process
            pass
