FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# ===== system deps =====
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    openslide-tools \
    libopenslide-dev \
    && rm -rf /var/lib/apt/lists/*

# ===== python setup =====
# 確保 python 指令存在
RUN ln -sf /usr/bin/python3 /usr/bin/python

# 升級 pip
RUN python -m pip install --upgrade pip

# ===== install core packages =====
RUN python -m pip install \
    pillow \
    tqdm \
    matplotlib \
    pandas \
    opencv-python \
    openslide-python \
    ultralytics==8.2.0 \
    openai \
    google-generativeai \
    reportlab

# 🔥 最重要：最後鎖 numpy（避免被升到 2.x）
RUN python -m pip install numpy==1.26.4


CMD ["/bin/bash"]