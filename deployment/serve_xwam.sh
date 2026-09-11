#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XWAM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXP_PATH="${EXP_PATH:-${XWAM_ROOT}/experiments/multitask_merged-v1-sft}"
WAN_CHECKPOINT_DIR="${WAN_CHECKPOINT_DIR:-${XWAM_ROOT}/checkpoints/Wan2.2-TI2V-5B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
STEPS="${STEPS:-last}"
DEPLOYMENT_CHECKPOINT="${DEPLOYMENT_CHECKPOINT:-${EXP_PATH}/checkpoints/${STEPS}.deployment.pt}"
PROFILE="${PROFILE:-0}"

[[ -x "${XWAM_ROOT}/.venv/bin/python" ]] || {
    echo "Missing X-WAM virtual environment: ${XWAM_ROOT}/.venv/bin/python" >&2
    exit 1
}
[[ -f "${EXP_PATH}/config.yaml" ]] || { echo "Missing experiment config: ${EXP_PATH}/config.yaml" >&2; exit 1; }
[[ -d "${WAN_CHECKPOINT_DIR}" ]] || { echo "Missing Wan checkpoint directory: ${WAN_CHECKPOINT_DIR}" >&2; exit 1; }
[[ -f "${DEPLOYMENT_CHECKPOINT}" ]] || {
    echo "Missing deployment checkpoint: ${DEPLOYMENT_CHECKPOINT}" >&2
    echo "Generate it with deployment/export_deployment_checkpoint.py first." >&2
    exit 1
}

"${XWAM_ROOT}/.venv/bin/python" -c 'import msgpack, websockets' || {
    echo "Missing WebSocket dependencies. Run: uv lock && uv sync" >&2
    exit 1
}

cd "${XWAM_ROOT}"
SERVER_ARGS=()
if [[ "${PROFILE}" == "1" ]]; then
    SERVER_ARGS+=(--profile)
fi
exec .venv/bin/python deployment/websocket_policy_server.py \
    --exp-path "${EXP_PATH}" \
    --wan-checkpoint-dir "${WAN_CHECKPOINT_DIR}" \
    --deployment-checkpoint "${DEPLOYMENT_CHECKPOINT}" \
    --steps "${STEPS}" --host "${HOST}" --port "${PORT}" "${SERVER_ARGS[@]}" "$@"
