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

COPY clearml/clearml.conf /root/clearml.conf

ENTRYPOINT ["clearml-agent"]
