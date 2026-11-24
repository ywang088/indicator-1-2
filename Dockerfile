FROM python:3.11.11-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY code/requirements-replay.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY code/ /workspace/code/
COPY scripts/ /workspace/scripts/
CMD ["bash", "scripts/01_build_18.sh"]
