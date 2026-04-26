#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
LOCK_FILE="${PROJECT_DIR}/.demo.lock"

cd "${PROJECT_DIR}"

if [ -f "${LOCK_FILE}" ]; then
  old_pid="$(tr -d ' \n\r' < "${LOCK_FILE}" || true)"
  if [ -n "${old_pid}" ] && kill -0 "${old_pid}" 2>/dev/null; then
    echo ">>> 检测到已有 demo 正在运行（PID=${old_pid}）"
    echo ">>> 请先在原终端输入 退出，再重新启动。"
    exit 1
  fi
fi

echo "$$" > "${LOCK_FILE}"
cleanup_lock() {
  rm -f "${LOCK_FILE}" 2>/dev/null || true
}
trap cleanup_lock EXIT INT TERM

if [ ! -d "${VENV_DIR}" ]; then
  echo ">>> 创建虚拟环境 .venv"
  python3 -m venv .venv
fi

echo ">>> 激活虚拟环境"
source "${VENV_DIR}/bin/activate"

if [ "${SKIP_INSTALL:-0}" = "1" ]; then
  echo ">>> 跳过依赖安装（SKIP_INSTALL=1）"
else
  echo ">>> 安装/更新依赖"
  python -m pip install --default-timeout=200 -r requirements.txt
fi

if [ -f "${PROJECT_DIR}/.env" ]; then
  echo ">>> 检测到 .env，安全加载环境变量"
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    if [[ -z "$line" ]] || [[ "$line" == \#* ]]; then
      continue
    fi
    if [[ "$line" != *=* ]]; then
      continue
    fi
    key="${line%%=*}"
    value="${line#*=}"
    key="${key%"${key##*[![:space:]]}"}"
    key="${key#"${key%%[![:space:]]*}"}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    value="${value%\"}"
    value="${value#\"}"
    value="${value%\'}"
    value="${value#\'}"
    if [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
      export "$key=$value"
    fi
  done < "${PROJECT_DIR}/.env"
fi

if [ -n "${DEEPSEEK_API_KEY:-}" ]; then
  echo ">>> 已检测到 DEEPSEEK_API_KEY（来源：shell 环境或 .env）"
else
  echo ">>> 未检测到 DEEPSEEK_API_KEY（shell 和 .env 都为空）"
  echo ">>> 请在 .env 设置：DEEPSEEK_API_KEY=你的key"
  exit 1
fi

echo ">>> 启动 Milvus LangChain 终端问答 Demo"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::UserWarning:milvus_lite}"
export GRPC_VERBOSITY="${GRPC_VERBOSITY:-ERROR}"
export GLOG_minloglevel="${GLOG_minloglevel:-2}"
python main.py
