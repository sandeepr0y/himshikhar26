# Step 1: Use a slim Python base image to start small (~150MB)
FROM nvcr.io/nvidia/pytorch:24.01-py3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Kolkata

WORKDIR /app

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata && \
    apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    python3-tk \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libx11-dev \
    libxext-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install opencv-python-headless==4.9.0.80 opencv-fixer==0.2.5 && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"

RUN pip install --no-cache-dir matplotlib --upgrade-strategy only-if-needed

CMD ["tail", "-f", "/dev/null"]