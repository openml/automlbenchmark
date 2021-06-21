import signal
import time
from unittest.mock import Mock

import pytest

from amlb.utils.process import InterruptTimeout, signal_handler
from amlb.utils.time import Timer


@pytest.mark.requires_unixlike
def test_interruption_behaves_like_a_keyboard_interruption_by_default():
    timeout = 1
    with Timer() as t:
        with pytest.raises(KeyboardInterrupt):
            with InterruptTimeout(timeout_secs=timeout):
                for i in range(100):
                    time.sleep(.1)
    assert t.duration - timeout < 1
    assert i < 11


def test_interruption_with_sig_as_None():
    timeout = 1
    with Timer() as t:
        with InterruptTimeout(timeout_secs=timeout, sig=None):
            for i in range(100):
                time.sleep(.1)
    assert t.duration - timeout < 1
    assert i < 11


def test_interruption_with_sig_as_error_class():
    timeout = 1
    with Timer() as t:
        with pytest.raises(TimeoutError, match=r"Interrupting thread (.*) after 1s timeout."):
            with InterruptTimeout(timeout_secs=timeout, sig=TimeoutError):
                for i in range(100):
                    time.sleep(.1)
    assert t.duration - timeout < 1
    assert i < 11


def test_interruption_with_sig_as_error_instance():
    timeout = 1
    with Timer() as t:
        with pytest.raises(TimeoutError, match=r"user provided error"):
            with InterruptTimeout(timeout_secs=timeout, sig=TimeoutError("user provided error")):
                for i in range(100):
                    time.sleep(.1)
    assert t.duration - timeout < 1
    assert i < 11


@pytest.mark.requires_unixlike
def test_interruption_with_sig_as_signal():
    def _handler(*_):
        raise TimeoutError("from handler")
    with signal_handler(signal.SIGTERM, _handler):
        timeout = 1
        with Timer() as t:
            with pytest.raises(TimeoutError, match=r"from handler"):
                with InterruptTimeout(timeout_secs=timeout, sig=signal.SIGTERM):
                    for i in range(100):
                        time.sleep(.1)
        assert t.duration - timeout < 1
        assert i < 11


def test_before_interrupt_hook():
    before = Mock()
    with Timer() as t:
        with InterruptTimeout(timeout_secs=1, sig=None, before_interrupt=before):
            for i in range(100):
                time.sleep(.1)
    assert before.called


@pytest.mark.requires_unixlike
def test_interruptions_escalation():
    def _handler(*_):
        raise TimeoutError("from handler")
    with signal_handler(signal.SIGINT, lambda *_: 0), signal_handler(signal.SIGTERM, _handler):
        before = Mock()

        timeout = 1
        with Timer() as t:
            with pytest.raises(TimeoutError, match=r"from handler"):
                with InterruptTimeout(timeout_secs=timeout,
                                      interruptions=[
                                          dict(sig=signal.SIGINT),
                                          dict(sig=signal.SIGTERM)
                                      ],
                                      before_interrupt=before):
                    for i in range(100):
                        time.sleep(.1)
        assert t.duration - timeout < 2  # default wait_retry_secs is 1s
        assert 15 < i < 25
        assert before.call_count == 2


@pytest.mark.requires_unixlike
def test_wait_retry_in_interruptions_escalation():
    def _handler(*_):
        raise TimeoutError("from handler")
    with signal_handler(signal.SIGINT, lambda *_: 0), signal_handler(signal.SIGTERM, _handler):
        timeout = 1
        with Timer() as t:
            with pytest.raises(TimeoutError, match=r"from handler"):
                with InterruptTimeout(timeout_secs=timeout,
                                      interruptions=[
                                          dict(sig=signal.SIGINT),
                                          dict(sig=signal.SIGTERM)
                                      ],
                                      wait_retry_secs=0.3):
                    for i in range(100):
                        time.sleep(.1)
        assert t.duration - timeout < 1
        assert 10 < i < 15

