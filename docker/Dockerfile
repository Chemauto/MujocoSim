ARG BASE_IMAGE=ros:jazzy-ros-base
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        python3-pip \
        libgl1 \
        libegl1 \
        libgles2 \
        mesa-utils \
        libglfw3 \
        libx11-6 \
        libxext6 \
        libxrandr2 \
        libxinerama1 \
        libxcursor1 \
        libxi6 \
        libxkbcommon0 \
        libxcb-cursor0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxcb-shape0 \
        libxcb-xinerama0 \
        libxcb-xkb1 \
        libxkbcommon-x11-0 \
        libdbus-1-3 \
        libfontconfig1 \
        fonts-noto-cjk \
        ros-jazzy-rviz2 \
    && rm -rf /var/lib/apt/lists/*

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install --break-system-packages --no-cache-dir \
        --index-url ${PIP_INDEX_URL} \
        mujoco \
        "numpy<2" \
        pyyaml \
        onnxruntime \
        opencv-python-headless \
        PySide6 \
        pygame

WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE=1
