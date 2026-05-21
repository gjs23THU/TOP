from __future__ import annotations

import contextlib
from dataclasses import asdict, is_dataclass
import logging
from pathlib import Path
from typing import Any

from .storage import read_result_json


logger = logging.getLogger("top_algorithm_service.solver")


class _LogStream:
    """File-like stream that forwards complete lines to the service logger."""

    def __init__(self, run_id: str | None, level: int) -> None:
        self.run_id = run_id
        self.level = level
        self._buffer = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._log(line)
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self._log(self._buffer)
            self._buffer = ""

    def _log(self, line: str) -> None:
        line = line.rstrip()
        if not line:
            return
        prefix = f"[run_id={self.run_id}] " if self.run_id else ""
        logger.log(self.level, "%s%s", prefix, line)


class SolverAdapter:
    """Small adapter around the internal solver package.

    Keeping this as the only import point for `unified_framework` makes the HTTP
    service independent from algorithm internals. Replacing the algorithm later
    should only require another adapter with the same `run` method.
    """

    def run(self, case_dir: Path, run_id: str | None = None) -> dict[str, Any]:
        from unified_framework.router import run_case

        logger.info("[run_id=%s] solver started case_dir=%s", run_id, case_dir)
        stdout = _LogStream(run_id, logging.INFO)
        stderr = _LogStream(run_id, logging.WARNING)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = run_case(case_dir)
        stdout.flush()
        stderr.flush()
        result_payload = read_result_json(case_dir)
        if result_payload is not None:
            logger.info(
                "[run_id=%s] solver finished status=%s objective=%s",
                run_id,
                result_payload.get("status"),
                result_payload.get("objective_value"),
            )
            if result_payload.get("error"):
                logger.info("[run_id=%s] solver error=%s", run_id, result_payload["error"])
            return result_payload
        if is_dataclass(result):
            payload = asdict(result)
            logger.info(
                "[run_id=%s] solver finished status=%s objective=%s",
                run_id,
                payload.get("status"),
                payload.get("objective_value"),
            )
            return payload
        raise RuntimeError("Solver did not produce a result payload")
