#!/usr/bin/env bash
set -euo pipefail

PORT="${SERVING_PORT:-${CLEARML_SERVING_PORT:-8080}}"
LOG_LEVEL="${UVICORN_LOG_LEVEL:-warning}"
LOOP="${UVICORN_SERVE_LOOP:-uvloop}"
EXTRA_ARGS="${UVICORN_EXTRA_ARGS:-}"

exec uvicorn clearml_serving.serving.main:app \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --log-level "${LOG_LEVEL}" \
  --loop "${LOOP}" \
  ${EXTRA_ARGS}
