# syntax=docker/dockerfile:1.4
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       libgomp1 \
       git \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
    "clearml==2.1.1" \
    "clearml-serving==1.3.5" \
    "fastapi==0.115.0" \
    "uvicorn[standard]==0.30.6" \
    "grpcio==1.66.1" \
    "boto3==1.41.5" \
    "kafka-python==2.0.2" \
    "lz4==4.3.3" \
    "numpy==1.26.4" \
    "pandas==2.2.3" \
    "autogluon[all]==1.4.0"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["/entrypoint.sh"]
