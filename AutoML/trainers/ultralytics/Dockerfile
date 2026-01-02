# syntax=docker/dockerfile:1.4
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install system deps and python (include OpenCV libs to avoid ClearML agent apt installs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    git \
    curl \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Ensure pip refers to python3
RUN ln -s /usr/bin/python3 /usr/bin/python || true

WORKDIR /app

COPY infra/clearml/clearml.conf /etc/clearml/clearml.conf
COPY infra/clearml/storage_credentials.conf /etc/clearml/storage_credentials.conf
COPY trainers/ultralytics/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

COPY shared_lib /app/shared_lib
COPY trainers/ultralytics/train.py /app/train.py

ENV CLEARML_CONFIG_FILE=/etc/clearml/clearml.conf
ENV TRAINS_CONFIG_FILE=/etc/clearml/clearml.conf

CMD ["python", "/app/train.py"]
