FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TOP_SERVICE_RUN_ROOT=/data/runs

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY algorithm_service ./algorithm_service
COPY unified_framework ./unified_framework

RUN mkdir -p /data/runs

EXPOSE 8000

CMD ["uvicorn", "algorithm_service.app:app", "--host", "0.0.0.0", "--port", "8000"]
