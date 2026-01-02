# syntax=docker/dockerfile:1.4
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       libgomp1 \
       git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY infra/clearml/clearml.conf /etc/clearml/clearml.conf
COPY infra/clearml/storage_credentials.conf /etc/clearml/storage_credentials.conf
COPY trainers/flaml/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

COPY shared_lib /app/shared_lib
COPY trainers/flaml/train.py /app/train.py

ENV CLEARML_CONFIG_FILE=/etc/clearml/clearml.conf
ENV TRAINS_CONFIG_FILE=/etc/clearml/clearml.conf

CMD ["python", "/app/train.py"]
