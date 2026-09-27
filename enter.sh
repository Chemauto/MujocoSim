#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "🤖 进入 MujocoSim 容器 ..."
echo "📂 项目目录: $(pwd)（已挂载到 /workspace，本地改动即时生效）"

if [[ -n "${DISPLAY:-}" ]]; then
    xhost +local:docker >/dev/null 2>&1 || true
fi

RUN_FLAGS=(--rm -i)
if [[ -t 0 && -t 1 ]]; then
    RUN_FLAGS+=(-t)
fi

exec docker run "${RUN_FLAGS[@]}" \
    --device nvidia.com/gpu=all \
    --network host \
    --ipc host \
    -v /dev/dri:/dev/dri \
    -e "DISPLAY=${DISPLAY:-:1}" \
    -e "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}" \
    -e "PYTHONDONTWRITEBYTECODE=1" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$PWD":/workspace \
    -w /workspace \
    mujocosim:latest "$@"
