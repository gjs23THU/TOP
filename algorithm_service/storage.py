from __future__ import annotations

import csv
import json
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import BinaryIO

from fastapi import HTTPException, UploadFile


REQUIRED_CASE_FILES = (
    "config.json",
    "task.csv",
    "package.csv",
    "point.csv",
    "distance.csv",
    "time.csv",
    "power.csv",
)


def create_case_dir(run_root: Path) -> tuple[str, Path]:
    run_root.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    case_dir = run_root / run_id
    case_dir.mkdir()
    return run_id, case_dir


def resolve_run_dir(run_root: Path, run_id: str) -> Path:
    if not run_id or any(char not in "0123456789abcdef" for char in run_id) or len(run_id) != 32:
        raise HTTPException(status_code=404, detail="Run not found")
    case_dir = run_root / run_id
    if not case_dir.exists() or not case_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")
    return case_dir


async def save_upload(upload: UploadFile, destination: Path, max_bytes: int) -> None:
    total = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"Uploaded file is larger than {max_bytes} bytes",
                )
            output.write(chunk)
    await upload.close()


async def save_required_uploads(
    case_dir: Path,
    uploads: dict[str, UploadFile],
    max_bytes: int,
) -> None:
    missing = [file_name for file_name in REQUIRED_CASE_FILES if file_name not in uploads]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required files: {missing}")
    for file_name in REQUIRED_CASE_FILES:
        await save_upload(uploads[file_name], case_dir / file_name, max_bytes)


def _safe_zip_member_name(name: str) -> bool:
    path = Path(name)
    return not path.is_absolute() and ".." not in path.parts and not name.endswith("/")


def extract_case_zip(
    archive: BinaryIO,
    case_dir: Path,
    max_member_bytes: int,
) -> None:
    try:
        with zipfile.ZipFile(archive) as zip_file:
            members = [info for info in zip_file.infolist() if not info.is_dir()]
            by_basename: dict[str, zipfile.ZipInfo] = {}
            for info in members:
                if not _safe_zip_member_name(info.filename):
                    raise HTTPException(status_code=400, detail="Zip archive contains unsafe paths")
                if info.file_size > max_member_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Zip member {info.filename} is larger than {max_member_bytes} bytes",
                    )
                basename = Path(info.filename).name
                if basename in REQUIRED_CASE_FILES:
                    if basename in by_basename:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Zip archive contains duplicate {basename}",
                        )
                    by_basename[basename] = info

            missing = [file_name for file_name in REQUIRED_CASE_FILES if file_name not in by_basename]
            if missing:
                raise HTTPException(status_code=422, detail=f"Missing required files: {missing}")

            for file_name, info in by_basename.items():
                with zip_file.open(info) as source, (case_dir / file_name).open("wb") as output:
                    shutil.copyfileobj(source, output)
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail="Invalid zip archive") from exc


def read_result_json(case_dir: Path) -> dict | None:
    result_path = case_dir / "output" / "result.json"
    if not result_path.exists():
        return None
    return json.loads(result_path.read_text(encoding="utf-8"))


def read_schedule_rows(case_dir: Path) -> list[dict[str, str]] | None:
    schedule_path = case_dir / "output" / "schedule.csv"
    if not schedule_path.exists():
        return None
    with schedule_path.open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))

