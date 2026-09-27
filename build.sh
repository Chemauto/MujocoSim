#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PROJECT_DIR="$(pwd)"
IMAGE="mujocosim:latest"
BASE_IMAGE="${BASE_IMAGE:-ros:jazzy-ros-base}"

echo "🤖 ================================"
echo "   MujocoSim · 镜像构建"
echo "================================"
echo "📂 项目目录 : ${PROJECT_DIR}"
echo "🐳 镜像名称 : ${IMAGE}"
echo "🔧 基础镜像 : ${BASE_IMAGE}"
echo

echo "📦 步骤 1/2：构建 Docker 镜像 ..."
if ! docker build --build-arg "BASE_IMAGE=${BASE_IMAGE}" -t "${IMAGE}" -f docker/Dockerfile .; then
    echo "⚠️  构建失败（网络波动？），自动换镜像加速站重试 ..."
    BASE_IMAGE="docker.m.daocloud.io/library/ros:jazzy-ros-base"
    docker build --build-arg "BASE_IMAGE=${BASE_IMAGE}" -t "${IMAGE}" -f docker/Dockerfile .
fi
echo "✅ 镜像构建完成：${IMAGE}（base: ${BASE_IMAGE}）"
echo

echo "🚀 步骤 2/2：打开 VSCode Dev Container ..."
HEX_PATH="$(printf '%s' "${PROJECT_DIR}" | xxd -p | tr -d '\n')"
if command -v code >/dev/null 2>&1; then
    code --folder-uri "vscode-remote://dev-container+${HEX_PATH}" >/dev/null 2>&1 \
        || code "${PROJECT_DIR}" >/dev/null 2>&1
    echo "✅ 已打开 VSCode（若窗口弹出后提示选择，选 Reopen in Container）"
else
    echo "⚠️  未找到 code 命令：请用 VSCode 打开 ${PROJECT_DIR}，再点 Reopen in Container"
fi
echo
echo "💡 进入容器终端后运行：python3 sim_entry.py   （打开仿真器）"
echo "💡 控制面板：python3 scripts/mainctl.py       虚拟手柄：python3 scripts/joystick_node.py"
