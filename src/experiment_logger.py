"""
Experiment Logger - Full LLM I/O Logging for Research Transparency

The active logger is exposed via a ``contextvars.ContextVar`` so each
thread / worker task has its own logger, and nested code (e.g. LangGraph
nodes) can reach it via :func:`get_logger` without explicit plumbing.
"""

import contextvars
import json
import os
import threading
from contextlib import contextmanager
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class LLMCallRecord:
    timestamp: str
    pipeline: str
    persona: str
    period: str
    step: str
    run: int
    system_prompt: str
    human_prompt: str
    raw_output: str
    parsed_output: Any
    success: bool
    error: str = ""
    model: str = ""
    temperature: float = 0.2
    # chars/4 heuristic — avoids pulling a tokenizer dep for a non-OpenAI model
    input_tokens: int = 0
    output_tokens: int = 0


class ExperimentLogger:
    def __init__(self, log_path="results/experiment_log.json", model: str = ""):
        self.log_path = log_path
        self.model = model
        self.records = []
        self._lock = threading.Lock()
        self._local = threading.local()

    def _ensure_local(self):
        if not hasattr(self._local, 'run'):
            self._local.run = 1
            self._local.pipeline = "single_agent"
            self._local.persona = ""
            self._local.period = ""

    def set_context(self, pipeline=None, persona=None, period=None, run=None):
        self._ensure_local()
        if pipeline is not None: self._local.pipeline = pipeline
        if persona is not None: self._local.persona = persona
        if period is not None: self._local.period = period
        if run is not None: self._local.run = run

    def log_llm_call(self, step, system_prompt, human_prompt, raw_output,
                     parsed_output=None, success=True, error="",
                     temperature=0.2, pipeline=None, persona=None,
                     period=None, run=None):
        input_tokens = (len(system_prompt) + len(human_prompt)) // 4
        output_tokens = len(raw_output) // 4 if raw_output else 0
        self._ensure_local()
        record = LLMCallRecord(
            timestamp=datetime.now().isoformat(),
            pipeline=pipeline or getattr(self._local, 'pipeline', 'single_agent'),
            persona=persona or getattr(self._local, 'persona', ''),
            period=period or getattr(self._local, 'period', ''),
            run=run or getattr(self._local, 'run', 1),
            step=step,
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            raw_output=raw_output,
            parsed_output=parsed_output,
            success=success, error=error,
            model=self.model,
            temperature=temperature,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        with self._lock:
            self.records.append(record)

    def log_event(self, event_type, data, pipeline=None, persona=None, period=None):
        self._ensure_local()
        record = LLMCallRecord(
            timestamp=datetime.now().isoformat(),
            pipeline=pipeline or getattr(self._local, 'pipeline', 'single_agent'),
            persona=persona or getattr(self._local, 'persona', ''),
            period=period or getattr(self._local, 'period', ''),
            run=getattr(self._local, 'run', 1),
            step="EVENT:" + event_type,
            system_prompt="", human_prompt="", raw_output="",
            parsed_output=data, success=True,
            model=self.model,
        )
        with self._lock:
            self.records.append(record)

    def save(self):
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        llm_calls = [r for r in self.records if not r.step.startswith("EVENT:")]
        output = {
            "experiment_log": {
                "model": self.model,
                "total_llm_calls": len(llm_calls),
                "total_events": len(self.records) - len(llm_calls),
                "total_records": len(self.records),
                "success_rate": sum(1 for r in llm_calls if r.success) / max(1, len(llm_calls)),
                "approx_total_input_tokens": sum(r.input_tokens for r in self.records),
                "approx_total_output_tokens": sum(r.output_tokens for r in self.records),
            },
            "records": [asdict(r) for r in self.records],
        }
        with open(self.log_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"  Log saved: {self.log_path} ({len(self.records)} records)")

    def print_summary(self):
        llm_calls = [r for r in self.records if not r.step.startswith("EVENT:")]
        failures = [r for r in llm_calls if not r.success]
        if failures:
            for f in failures:
                print(f"    {f.persona}/{f.period}/{f.step}: {f.error[:80]}")


# ── Thread-/task-local active logger ─────────────────────────────────────────
_active_logger: contextvars.ContextVar[Optional[ExperimentLogger]] = contextvars.ContextVar(
    "experiment_logger", default=None
)


def get_logger() -> Optional[ExperimentLogger]:
    return _active_logger.get()


def set_logger(logger: Optional[ExperimentLogger]):
    """Bind logger for the current context. Returns a reset token."""
    return _active_logger.set(logger)


@contextmanager
def with_logger(logger: Optional[ExperimentLogger]):
    """Scope a logger to a block — restores the previous binding on exit."""
    token = _active_logger.set(logger)
    try:
        yield logger
    finally:
        _active_logger.reset(token)
