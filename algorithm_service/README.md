# TOP Algorithm FastAPI Service

This package is an HTTP wrapper around the existing case-directory interface.
It intentionally keeps the service layer independent from the algorithm
implementation:

- HTTP, upload, run-directory, and response logic live in `algorithm_service`.
- The only algorithm dependency is `SolverAdapter.run(case_dir)`, which calls
  `unified_framework.router.run_case`.
- The algorithm input and output files remain the documented interface:
  `config.json`, `task.csv`, `package.csv`, `point.csv`, `distance.csv`,
  `time.csv`, `power.csv`, `output/result.json`, and `output/schedule.csv`.

## Install service dependencies

```bash
./.venv/bin/pip install -r requirements-service.txt
```

## Run

```bash
./.venv/bin/uvicorn algorithm_service.app:app --host 0.0.0.0 --port 8000
```

Run directories default to `/tmp/top_algorithm_service_runs`. Override with:

```bash
TOP_SERVICE_RUN_ROOT=/path/to/runs ./.venv/bin/uvicorn algorithm_service.app:app --host 0.0.0.0 --port 8000
```

## API

### Health

```bash
curl http://127.0.0.1:8000/health
```

### Solve from multipart files

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/solve" \
  -F "config=@unified_framework/inputs/instance1/config.json" \
  -F "task=@unified_framework/inputs/instance1/task.csv" \
  -F "package=@unified_framework/inputs/instance1/package.csv" \
  -F "point=@unified_framework/inputs/instance1/point.csv" \
  -F "distance=@unified_framework/inputs/instance1/distance.csv" \
  -F "time=@unified_framework/inputs/instance1/time.csv" \
  -F "power=@unified_framework/inputs/instance1/power.csv"
```

### Solve from a zip

The zip may contain the seven required files at the root or inside one case
folder. Duplicate required basenames are rejected.

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/solve-zip" \
  -F "file=@case_001.zip"
```

### Retrieve outputs

Use the `run_id` returned by a solve request:

```bash
curl http://127.0.0.1:8000/api/v1/runs/<run_id>/result
curl http://127.0.0.1:8000/api/v1/runs/<run_id>/schedule
curl -O http://127.0.0.1:8000/api/v1/runs/<run_id>/schedule.csv
```

