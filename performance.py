"""Request timings without logging questions, photos, answers or credentials."""

import logging
import math
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

logger = logging.getLogger("terramind.performance")


@dataclass
class _Profile:
    stages: list = field(default_factory=list)


_current = ContextVar("request_performance", default=None)


@contextmanager
def request_profile(request_id, submitted_at):
    if os.getenv("PERFORMANCE_LOGGING", "1").strip().lower() in {"0", "false", "off"}:
        yield
        return

    started = time.monotonic()
    queued = max(0.0, started - submitted_at)
    profile = _Profile()
    token = _current.set(profile)
    status = "ok"
    try:
        yield
    except BaseException:
        status = "failed"
        raise
    finally:
        elapsed = time.monotonic() - started
        _current.reset(token)
        # Uvicorn configures its own loggers; make these diagnostics visible too.
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        prefix = f"[timing {request_id[:8]}]"
        for name, duration, details in profile.stages:
            extra = ", ".join(f"{key}={value:.3f}" for key, value in details.items())
            logger.info("%s %s=%.3fs%s", prefix, name, duration,
                        f" ({extra})" if extra else "")
        logger.info("%s queue=%.3fs processing=%.3fs total=%.3fs status=%s",
                    prefix, queued, elapsed, queued + elapsed, status)


@contextmanager
def timed_stage(name):
    profile = _current.get()
    if profile is None:
        yield None
        return
    started = time.monotonic()
    details = {}
    try:
        yield details
    finally:
        profile.stages.append((name, time.monotonic() - started, details))


def invoke_chat(model, messages, stage):
    with timed_stage(stage) as details:
        response = model.invoke(messages)
        metadata = getattr(response, "response_metadata", None)
        if details is not None and isinstance(metadata, dict):
            for key, label, scale in (
                ("load_duration", "load_s", 1e9),
                ("prompt_eval_duration", "prompt_s", 1e9),
                ("eval_duration", "generate_s", 1e9),
                ("prompt_eval_count", "input_tokens", 1),
                ("eval_count", "output_tokens", 1),
            ):
                value = metadata.get(key)
                if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                    details[label] = value / scale
        return response
