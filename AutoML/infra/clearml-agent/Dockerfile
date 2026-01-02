FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
       curl \
       git \
       docker.io \
       docker-cli \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir clearml-agent==2.0.6

COPY clearml/clearml.conf /etc/clearml/clearml.conf
COPY clearml/storage_credentials.conf /etc/clearml/storage_credentials.conf

ENV CLEARML_CONFIG_FILE=/etc/clearml/clearml.conf
ENV TRAINS_CONFIG_FILE=/etc/clearml/clearml.conf

ENTRYPOINT ["clearml-agent"]
