from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServiceSettings:
    """Runtime settings for the HTTP wrapper.

    The solver itself still reads and writes a case directory. The service keeps
    those directories outside the algorithm package so the HTTP boundary remains
    replaceable.
    """

    run_root: Path
    max_upload_bytes: int = 200 * 1024 * 1024
    max_zip_member_bytes: int = 200 * 1024 * 1024


def get_settings() -> ServiceSettings:
    run_root = Path(
        os.environ.get("TOP_SERVICE_RUN_ROOT", "/tmp/top_algorithm_service_runs")
    ).expanduser()
    max_upload_bytes = int(os.environ.get("TOP_SERVICE_MAX_UPLOAD_BYTES", 200 * 1024 * 1024))
    max_zip_member_bytes = int(
        os.environ.get("TOP_SERVICE_MAX_ZIP_MEMBER_BYTES", max_upload_bytes)
    )
    return ServiceSettings(
        run_root=run_root,
        max_upload_bytes=max_upload_bytes,
        max_zip_member_bytes=max_zip_member_bytes,
    )

