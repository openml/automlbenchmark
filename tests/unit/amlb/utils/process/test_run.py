import pytest

from amlb.utils import StaleProcessError, run_cmd


@pytest.mark.parametrize("mode", ["line", "block", True])
def test_subprocess_detects_stale_process(mode):
    one_ms = 0.001
    with pytest.raises(StaleProcessError):
        run_cmd(f"sleep {one_ms}", _activity_timeout_=one_ms / 2, _live_output_=mode)


def test_subprocess_does_not_raises_if_process_exits_early():
    run_cmd("echo hi", _activity_timeout_=1, _live_output_=True)


@pytest.mark.xfail(reason="Detection of stale processes currently requires output")
def test_subprocess_does_not_raises_on_silent_process():
    one_ms = 0.001
    run_cmd(f"sleep {one_ms}", _activity_timeout_=one_ms / 2, _live_output_=True)
