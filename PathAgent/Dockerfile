FROM python:3.9-slim

# 避免互動
ENV DEBIAN_FRONTEND=noninteractive

# 安裝系統套件（CLAM + OpenCV + openslide 會用到）
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    openslide-tools \
    libopenslide-dev \
    && rm -rf /var/lib/apt/lists/*

# 設定工作目錄
WORKDIR /workspace

# 安裝 Python 套件
RUN pip install --upgrade pip

# 先裝必要的（避免 cv2 error）
RUN pip install opencv-python-headless

# 如果你有 requirements.txt
# COPY requirements.txt .
# RUN pip install -r requirements.txt