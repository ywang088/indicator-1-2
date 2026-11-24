FROM --platform=linux/amd64 python:3.11.11-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 EASYOCR_MODEL_DIR=/workspace/data/models/ocr
RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu-core libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
RUN pip install --no-cache-dir torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130
COPY code/requirements-gpu.txt /tmp/requirements.txt
COPY code/vendor/sam3-0.1.4-py3-none-any.whl /tmp/sam3.whl
RUN pip install --no-cache-dir -r /tmp/requirements.txt && mv /tmp/sam3.whl /tmp/sam3-0.1.4-py3-none-any.whl && pip install --no-cache-dir /tmp/sam3-0.1.4-py3-none-any.whl
COPY code/ /workspace/code/
COPY scripts/ /workspace/scripts/
CMD ["bash", "scripts/01_build_18.sh"]
