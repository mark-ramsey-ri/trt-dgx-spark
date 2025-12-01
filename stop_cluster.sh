#!/usr/bin/env bash
set -euo pipefail

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TensorRT-LLM Cluster Shutdown Script
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Stops TensorRT-LLM containers on head and all worker nodes.
# Run from head node - it will SSH to workers automatically.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load configuration
if [ -f "${SCRIPT_DIR}/config.local.env" ]; then
  source "${SCRIPT_DIR}/config.local.env"
elif [ -f "${SCRIPT_DIR}/config.env" ]; then
  source "${SCRIPT_DIR}/config.env"
fi

HEAD_CONTAINER_NAME="${HEAD_CONTAINER_NAME:-trtllm-head}"
WORKER_CONTAINER_NAME="${WORKER_CONTAINER_NAME:-trtllm-worker}"
# Support both new (WORKER_HOST) and legacy (WORKER_IPS) variable names
WORKER_HOST="${WORKER_HOST:-${WORKER_IPS:-}}"
WORKER_USER="${WORKER_USER:-$(whoami)}"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

stop_local_container() {
  local name="$1"
  if docker ps -a --format '{{.Names}}' | grep -qx "${name}"; then
    log "  Stopping ${name}..."
    docker stop "${name}" >/dev/null 2>&1 || true
    docker rm -f "${name}" >/dev/null 2>&1 || true
    log "  ${name} stopped"
    return 0
  else
    return 1
  fi
}

stop_remote_containers() {
  local host="$1"
  local user="$2"

  log "  Stopping containers on ${host}..."

  if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "${user}@${host}" "echo ok" >/dev/null 2>&1; then
    log "  Warning: Cannot SSH to ${user}@${host}, skipping"
    return 1
  fi

  ssh "${user}@${host}" bash -s << 'REMOTE_EOF'
# Stop all trtllm-* containers on remote node
CONTAINERS=$(docker ps -a --format '{{.Names}}' | grep -E "^trtllm-" || true)
if [ -n "${CONTAINERS}" ]; then
  for c in ${CONTAINERS}; do
    docker stop "${c}" >/dev/null 2>&1 || true
    docker rm -f "${c}" >/dev/null 2>&1 || true
    echo "    Stopped ${c}"
  done
else
  echo "    No TensorRT-LLM containers found"
fi
REMOTE_EOF
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Parse Arguments
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FORCE=false
LOCAL_ONLY=false

while [[ $# -gt 0 ]]; do
  case $1 in
    -f|--force)
      FORCE=true
      shift
      ;;
    -l|--local-only)
      LOCAL_ONLY=true
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  -f, --force       Stop without confirmation"
      echo "  -l, --local-only  Only stop containers on this node (don't SSH to workers)"
      echo "  -h, --help        Show this help"
      echo ""
      echo "By default, this script will:"
      echo "  1. Stop containers on the head node (local)"
      echo "  2. SSH to all workers in WORKER_HOST and stop their containers"
      echo ""
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Script
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo ""
echo "============================================================="
echo " TensorRT-LLM Cluster Shutdown"
echo "============================================================="
echo ""

# Convert WORKER_HOST to array (for SSH access to workers)
read -ra WORKER_HOST_ARRAY <<< "${WORKER_HOST}"

# Show what will be stopped
log "Will stop TensorRT-LLM on:"
echo "  - Head node (local)"
if [ "${LOCAL_ONLY}" != "true" ] && [ ${#WORKER_HOST_ARRAY[@]} -gt 0 ]; then
  for ip in "${WORKER_HOST_ARRAY[@]}"; do
    echo "  - Worker: ${ip}"
  done
fi
echo ""

# Find local containers
LOCAL_CONTAINERS=$(docker ps --format '{{.Names}}' | grep -E "^trtllm-" || true)

if [ -z "${LOCAL_CONTAINERS}" ]; then
  log "No TensorRT-LLM containers on head node."
else
  log "Local containers:"
  for c in ${LOCAL_CONTAINERS}; do
    echo "  - ${c}"
  done
fi
echo ""

# Confirmation
if [ "${FORCE}" != "true" ]; then
  read -p "Proceed with shutdown? [y/N] " -n 1 -r
  echo ""
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log "Cancelled."
    exit 0
  fi
  echo ""
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Stop workers first (if configured)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if [ "${LOCAL_ONLY}" != "true" ] && [ ${#WORKER_HOST_ARRAY[@]} -gt 0 ]; then
  log "Stopping workers..."
  for ip in "${WORKER_HOST_ARRAY[@]}"; do
    stop_remote_containers "${ip}" "${WORKER_USER}" || true
  done
  echo ""
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Stop head node containers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log "Stopping head node..."
STOPPED=0

# Stop head container
if stop_local_container "${HEAD_CONTAINER_NAME}"; then
  STOPPED=$((STOPPED + 1))
fi

# Stop any local worker containers (shouldn't exist on head, but just in case)
for c in $(docker ps --format '{{.Names}}' | grep -E "^${WORKER_CONTAINER_NAME}" || true); do
  if stop_local_container "${c}"; then
    STOPPED=$((STOPPED + 1))
  fi
done

# Stop any other trtllm-* containers
for c in $(docker ps --format '{{.Names}}' | grep -E "^trtllm-" || true); do
  if stop_local_container "${c}"; then
    STOPPED=$((STOPPED + 1))
  fi
done

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Summary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo ""
echo "============================================================="
log "Cluster shutdown complete"
echo ""
echo "Stopped:"
echo "  - ${STOPPED} container(s) on head node"
if [ "${LOCAL_ONLY}" != "true" ] && [ ${#WORKER_HOST_ARRAY[@]} -gt 0 ]; then
  echo "  - Containers on ${#WORKER_HOST_ARRAY[@]} worker node(s)"
fi
echo ""
echo "============================================================="
