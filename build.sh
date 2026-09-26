#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
BASE_IMAGE="${BASE_IMAGE:-ros:jazzy-ros-base}"
docker build --build-arg "BASE_IMAGE=${BASE_IMAGE}" -t mujocosim:latest -f docker/Dockerfile .
echo "mujocosim:latest built (base: ${BASE_IMAGE})"
