#!/usr/bin/env bash
set -uo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
BASE_URL="${BASE_URL%/}"
INPUT_ZIP="${INPUT_ZIP:-instance1.zip}"
OUTPUT_DIR="${OUTPUT_DIR:-output/instance1_docker_matrix}"
TIME_LIMIT_SECONDS="${TIME_LIMIT_SECONDS:-120}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

ALGORITHMS=(${ALGORITHMS:-ha ga sa pso})
OBJECTIVES=(${OBJECTIVES:-maxRevenue minTime minPower})
BANNED_ALGORITHMS=" ea ai "
REQUIRED_FILES=(config.json task.csv package.csv point.csv distance.csv time.csv power.csv)

CSV_DIR="$OUTPUT_DIR/csv"
RESP_DIR="$OUTPUT_DIR/responses"
CASE_DIR="$OUTPUT_DIR/cases"
SUMMARY="$OUTPUT_DIR/summary.csv"

mkdir -p "$CSV_DIR" "$RESP_DIR" "$CASE_DIR"

if [[ ! -f "$INPUT_ZIP" ]]; then
  echo "Input zip not found: $INPUT_ZIP" >&2
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found: $PYTHON_BIN" >&2
  exit 1
fi

if ! curl -fsS "$BASE_URL/health" >/dev/null; then
  echo "Service health check failed: $BASE_URL/health" >&2
  exit 1
fi

echo "algorithm,objective,http_code,status,objective_value,run_id,schedule_file,error_type,error_message" > "$SUMMARY"

make_case_zip() {
  local algorithm="$1"
  local objective="$2"
  local output_zip="$3"

  "$PYTHON_BIN" - "$INPUT_ZIP" "$output_zip" "$algorithm" "$objective" "$TIME_LIMIT_SECONDS" <<'PY'
import json
import sys
import zipfile

input_zip, output_zip, algorithm, objective, time_limit = sys.argv[1:6]
required = ["config.json", "task.csv", "package.csv", "point.csv", "distance.csv", "time.csv", "power.csv"]

with zipfile.ZipFile(input_zip, "r") as source:
    files = {}
    for info in source.infolist():
        if info.is_dir():
            continue
        basename = info.filename.rsplit("/", 1)[-1]
        if basename in required:
            if basename in files:
                raise SystemExit(f"duplicate required file in zip: {basename}")
            files[basename] = source.read(info.filename)

missing = [name for name in required if name not in files]
if missing:
    raise SystemExit(f"missing required files: {missing}")

config = json.loads(files["config.json"].decode("utf-8"))
config.setdefault("algorithm", {})
config["algorithm"]["name"] = algorithm
config["algorithm"]["mode"] = "normal"
config["algorithm"]["obj"] = objective
if algorithm == "ha":
    config["algorithm"]["timeLimit"] = None
else:
    config["algorithm"]["timeLimit"] = int(time_limit)
if algorithm in {"ga", "sa", "pso"}:
    config["algorithm"]["random_seed"] = 42

files["config.json"] = json.dumps(config, ensure_ascii=False, indent=2).encode("utf-8")

with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as target:
    for name in required:
        target.writestr(f"instance1/{name}", files[name])
PY
}

parse_response() {
  local response_file="$1"
  "$PYTHON_BIN" - "$response_file" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    payload = json.loads(open(path, encoding="utf-8").read())
except Exception as exc:
    print("\t".join(["", "", "", "", "", "JSONDecodeError", str(exc).replace("\n", " ")]))
    raise SystemExit(0)

result = payload.get("result") or {}
error = result.get("error") or payload.get("error") or {}
print("\t".join([
    str(payload.get("status") or result.get("status") or ""),
    str(result.get("objective_value") if result.get("objective_value") is not None else ""),
    str(payload.get("run_id") or ""),
    str(payload.get("schedule_url") or ""),
    str(result.get("status") or ""),
    str(error.get("type") or ""),
    str(error.get("message") or "").replace("\n", " "),
]))
PY
}

append_summary() {
  local algorithm="$1"
  local objective="$2"
  local http_code="$3"
  local status="$4"
  local objective_value="$5"
  local run_id="$6"
  local schedule_file="$7"
  local error_type="$8"
  local error_message="$9"

  "$PYTHON_BIN" - "$SUMMARY" \
    "$algorithm" "$objective" "$http_code" "$status" "$objective_value" \
    "$run_id" "$schedule_file" "$error_type" "$error_message" <<'PY'
import csv
import sys

path = sys.argv[1]
row = sys.argv[2:]
with open(path, "a", encoding="utf-8", newline="") as output:
    csv.writer(output).writerow(row)
PY
}

for algorithm in "${ALGORITHMS[@]}"; do
  if [[ "$BANNED_ALGORITHMS" == *" $algorithm "* ]]; then
    echo "SKIP banned/unimplemented algorithm: $algorithm"
    continue
  fi

  for objective in "${OBJECTIVES[@]}"; do
    test_name="${algorithm}_${objective}"
    case_zip="$CASE_DIR/$test_name.zip"
    response_file="$RESP_DIR/$test_name.json"
    csv_file="$CSV_DIR/$test_name.csv"

    echo "RUN $test_name"
    if ! make_case_zip "$algorithm" "$objective" "$case_zip"; then
      echo "  failed to create case zip"
      append_summary "$algorithm" "$objective" "NA" "case_build_failed" "" "" "" "CaseBuildError" "failed to create case zip"
      continue
    fi

    http_code="$(
      curl -sS -X POST "$BASE_URL/api/v1/solve-zip?include_schedule=false" \
        -F "file=@$case_zip" \
        -o "$response_file" \
        -w "%{http_code}"
    )"

    if [[ "$http_code" != "200" ]]; then
      message="$(tr '\n' ' ' < "$response_file" 2>/dev/null || true)"
      echo "  HTTP $http_code"
      append_summary "$algorithm" "$objective" "$http_code" "http_error" "" "" "" "HTTPError" "$message"
      continue
    fi

    IFS=$'\t' read -r response_status objective_value run_id schedule_url result_status error_type error_message < <(parse_response "$response_file")
    status="${result_status:-$response_status}"
    downloaded=""

    if [[ "$status" == "success" && -n "$schedule_url" ]]; then
      if curl -fsS -o "$csv_file" "$BASE_URL$schedule_url"; then
        downloaded="$csv_file"
        echo "  success objective=$objective_value csv=$csv_file"
      else
        error_type="DownloadError"
        error_message="failed to download schedule csv from $schedule_url"
        echo "  success but csv download failed"
      fi
    else
      echo "  status=$status error=$error_type $error_message"
    fi

    append_summary "$algorithm" "$objective" "$http_code" "$status" "$objective_value" "$run_id" "$downloaded" "$error_type" "$error_message"
  done
done

echo "Done."
echo "Summary: $SUMMARY"
echo "CSV dir:  $CSV_DIR"
