from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorInfo(BaseModel):
    type: str
    message: str


class SolverResult(BaseModel):
    case_id: str
    status: str
    objective_value: float | None = None
    schedule_path: str | None = None
    error: ErrorInfo | None = None


class SolveResponse(BaseModel):
    run_id: str
    status: str
    result: SolverResult
    result_url: str
    schedule_url: str | None = None
    schedule: list[dict[str, Any]] | None = Field(
        default=None,
        description="Parsed schedule.csv rows when include_schedule=true and a schedule exists.",
    )


class RunSummary(BaseModel):
    run_id: str
    case_dir: str
    result: SolverResult | None = None


class HealthResponse(BaseModel):
    status: str
    solver: str
    run_root: str

