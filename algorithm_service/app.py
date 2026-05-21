from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from .schemas import HealthResponse, RunSummary, SolveResponse, SolverResult
from .settings import ServiceSettings, get_settings
from .solver_adapter import SolverAdapter
from .storage import (
    create_case_dir,
    extract_case_zip,
    read_result_json,
    read_schedule_rows,
    resolve_run_dir,
    save_required_uploads,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("top_algorithm_service")


def create_app(
    settings: ServiceSettings | None = None,
    solver: SolverAdapter | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    solver = solver or SolverAdapter()

    app = FastAPI(
        title="TOP Algorithm Service",
        version="1.0.0",
        description="Independent FastAPI wrapper for the TOP case-directory algorithm interface.",
    )

    def log_case_config(run_id: str, case_dir: Path) -> None:
        config_path = case_dir / "config.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            algorithm = config.get("algorithm", {})
            logger.info(
                "[run_id=%s] config loaded case_id=%s algorithm=%s mode=%s obj=%s timeLimit=%s",
                run_id,
                config.get("case_id", case_dir.name),
                algorithm.get("name"),
                algorithm.get("mode"),
                algorithm.get("obj"),
                algorithm.get("timeLimit"),
            )
        except Exception:
            logger.exception("[run_id=%s] failed to read config for progress logging", run_id)

    def build_response(
        run_id: str,
        case_dir: Path,
        result_payload: dict,
        include_schedule: bool,
    ) -> SolveResponse:
        schedule_rows = read_schedule_rows(case_dir) if include_schedule else None
        schedule_url = None
        if result_payload.get("schedule_path"):
            schedule_url = f"/api/v1/runs/{run_id}/schedule.csv"
        return SolveResponse(
            run_id=run_id,
            status=str(result_payload.get("status", "")),
            result=SolverResult(**result_payload),
            result_url=f"/api/v1/runs/{run_id}/result",
            schedule_url=schedule_url,
            schedule=schedule_rows,
        )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            solver="unified_framework",
            run_root=str(settings.run_root),
        )

    @app.get("/api/v1/algorithms")
    def algorithms() -> dict[str, list[str]]:
        return {
            "name": ["ea", "eao", "ha", "ga", "sa", "pso", "ai"],
            "mode": ["normal", "revisional", "back"],
            "objective": ["maxRevenue", "minTime", "minPower"],
        }

    @app.post("/api/v1/solve", response_model=SolveResponse)
    async def solve_multipart(
        config: UploadFile = File(...),
        task: UploadFile = File(...),
        package: UploadFile = File(...),
        point: UploadFile = File(...),
        distance: UploadFile = File(...),
        time: UploadFile = File(...),
        power: UploadFile = File(...),
        include_schedule: bool = Query(True),
    ) -> SolveResponse:
        run_id, case_dir = create_case_dir(settings.run_root)
        logger.info("[run_id=%s] received multipart solve request", run_id)
        uploads = {
            "config.json": config,
            "task.csv": task,
            "package.csv": package,
            "point.csv": point,
            "distance.csv": distance,
            "time.csv": time,
            "power.csv": power,
        }
        logger.info("[run_id=%s] saving uploaded case files", run_id)
        await save_required_uploads(case_dir, uploads, settings.max_upload_bytes)
        logger.info("[run_id=%s] uploaded case files saved", run_id)
        log_case_config(run_id, case_dir)
        result_payload = await run_in_threadpool(solver.run, case_dir, run_id)
        logger.info("[run_id=%s] building HTTP response", run_id)
        return build_response(run_id, case_dir, result_payload, include_schedule)

    @app.post("/api/v1/solve-zip", response_model=SolveResponse)
    async def solve_zip(
        file: UploadFile = File(...),
        include_schedule: bool = Query(True),
    ) -> SolveResponse:
        run_id, case_dir = create_case_dir(settings.run_root)
        logger.info("[run_id=%s] received zip solve request filename=%s", run_id, file.filename)
        try:
            logger.info("[run_id=%s] extracting zip upload", run_id)
            await run_in_threadpool(
                extract_case_zip,
                file.file,
                case_dir,
                settings.max_zip_member_bytes,
            )
            logger.info("[run_id=%s] zip extracted, required files ready", run_id)
        finally:
            await file.close()
        log_case_config(run_id, case_dir)
        result_payload = await run_in_threadpool(solver.run, case_dir, run_id)
        logger.info("[run_id=%s] building HTTP response", run_id)
        return build_response(run_id, case_dir, result_payload, include_schedule)

    @app.get("/api/v1/runs/{run_id}", response_model=RunSummary)
    def get_run(run_id: str) -> RunSummary:
        case_dir = resolve_run_dir(settings.run_root, run_id)
        result_payload = read_result_json(case_dir)
        return RunSummary(
            run_id=run_id,
            case_dir=str(case_dir),
            result=SolverResult(**result_payload) if result_payload else None,
        )

    @app.get("/api/v1/runs/{run_id}/result", response_model=SolverResult)
    def get_result(run_id: str) -> SolverResult:
        case_dir = resolve_run_dir(settings.run_root, run_id)
        result_payload = read_result_json(case_dir)
        if result_payload is None:
            raise HTTPException(status_code=404, detail="Result not found")
        return SolverResult(**result_payload)

    @app.get("/api/v1/runs/{run_id}/schedule")
    def get_schedule(run_id: str) -> list[dict[str, str]]:
        case_dir = resolve_run_dir(settings.run_root, run_id)
        rows = read_schedule_rows(case_dir)
        if rows is None:
            raise HTTPException(status_code=404, detail="Schedule not found")
        return rows

    @app.get("/api/v1/runs/{run_id}/schedule.csv")
    def download_schedule(run_id: str) -> FileResponse:
        case_dir = resolve_run_dir(settings.run_root, run_id)
        schedule_path = case_dir / "output" / "schedule.csv"
        if not schedule_path.exists():
            raise HTTPException(status_code=404, detail="Schedule not found")
        return FileResponse(
            schedule_path,
            media_type="text/csv",
            filename=f"{run_id}-schedule.csv",
        )

    return app


app = create_app()
