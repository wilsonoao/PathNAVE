FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV MAX_JOBS=4

# ===== system deps =====
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    openslide-tools \
    libopenslide-dev \
    && rm -rf /var/lib/apt/lists/*

# ===== python setup =====
RUN ln -sf /usr/bin/python3 /usr/bin/python

RUN python -m pip install --upgrade pip

# ===== core python packages =====
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
    reportlab \
    python-dotenv

# 🔥 鎖 numpy（避免 numpy 2.x 問題）
RUN python -m pip install numpy==1.26.4

# ===== transformers（放後面避免被覆蓋） =====
RUN python -m pip install \
    transformers \
    accelerate \
    qwen-vl-utils

# ===== FlashAttention（最後裝） =====
RUN python -m pip install flash-attn --no-build-isolation

CMD ["/bin/bash"]