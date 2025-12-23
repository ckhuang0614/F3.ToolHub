FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install system deps and python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    git \
    curl \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Ensure pip refers to python3
RUN ln -s /usr/bin/python3 /usr/bin/python || true

WORKDIR /app

COPY infra/clearml/clearml.conf /root/clearml.conf
COPY trainers/yolo/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY shared_lib /app/shared_lib
COPY trainers/yolo/train.py /app/train.py

ENV CLEARML_CONFIG_FILE=/root/clearml.conf

CMD ["python", "/app/train.py"]
