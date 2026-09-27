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
        libgl1-mesa-dri \
        libglx-mesa0 \
        libegl-mesa0 \
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
        zsh \
        curl \
        python-is-python3 \
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

RUN curl -fsSL --connect-timeout 15 -o /tmp/omz-install.sh \
        https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh \
    || curl -fsSL --connect-timeout 15 -o /tmp/omz-install.sh \
        https://gitee.com/mirrors/oh-my-zsh/raw/master/tools/install.sh

RUN sh /tmp/omz-install.sh "" --unattended \
    || { git clone --depth=1 https://gitee.com/mirrors/oh-my-zsh.git /root/.oh-my-zsh \
         && cp /root/.oh-my-zsh/templates/zshrc.zsh-template /root/.zshrc; }

COPY docker/zshrc /root/.zshrc
COPY docker/patch_resmon.py /tmp/patch_resmon.py

RUN curl -fsSL -o /tmp/rm.vsix "https://mutantdino.gallery.vsassets.io/_apis/public/gallery/publisher/mutantdino/extension/resourcemonitor/latest/assetbyname/Microsoft.VisualStudio.Services.VSIXPackage" && \
    python3 -m zipfile -e /tmp/rm.vsix /tmp/rmext && \
    python3 /tmp/patch_resmon.py && \
    mkdir -p /root/.vscode-server/extensions && \
    mv /tmp/rmext/extension /root/.vscode-server/extensions/mutantdino.resourcemonitor-1.0.7 && \
    rm -rf /tmp/rm.vsix /tmp/rmext /tmp/patch_resmon.py

RUN echo 'export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}' >> /root/.bashrc && \
    echo 'export DISPLAY=${DISPLAY:-:1}' >> /root/.bashrc && \
    echo '[[ -f /opt/ros/jazzy/setup.bash ]] && source /opt/ros/jazzy/setup.bash' >> /root/.bashrc && \
    echo '[[ $PWD == / ]] && cd /workspace' >> /root/.bashrc && \
    { grep -qx "$(command -v zsh)" /etc/shells || echo "$(command -v zsh)" >> /etc/shells; } && \
    chsh -s "$(command -v zsh)" root

CMD ["zsh"]
