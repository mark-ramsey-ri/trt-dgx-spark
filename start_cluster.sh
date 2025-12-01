#!/usr/bin/env bash
set -euo pipefail

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TensorRT-LLM DGX Spark Cluster - Unified Start Script
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Starts TensorRT-LLM on both head and worker nodes from a single command.
# Run this script on the HEAD NODE - it will SSH to workers automatically.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load configuration
if [ -f "${SCRIPT_DIR}/config.local.env" ]; then
  source "${SCRIPT_DIR}/config.local.env"
elif [ -f "${SCRIPT_DIR}/config.env" ]; then
  source "${SCRIPT_DIR}/config.env"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration with defaults
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Docker
TRT_IMAGE="${TRT_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:1.0.0rc3}"
HEAD_CONTAINER_NAME="${HEAD_CONTAINER_NAME:-trtllm-head}"
WORKER_CONTAINER_NAME="${WORKER_CONTAINER_NAME:-trtllm-worker}"
SHM_SIZE="${SHM_SIZE:-32g}"

# Model
MODEL="${MODEL:-nvidia/Qwen3-235B-A22B-FP4}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-2}"
NUM_NODES="${NUM_NODES:-2}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-4}"
MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-32768}"

# TensorRT-LLM options
TRT_BACKEND="${TRT_BACKEND:-pytorch}"
GPU_MEMORY_FRACTION="${GPU_MEMORY_FRACTION:-0.90}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
CUDA_GRAPH_PADDING="${CUDA_GRAPH_PADDING:-true}"
DISABLE_OVERLAP_SCHEDULER="${DISABLE_OVERLAP_SCHEDULER:-true}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Ports
TRT_PORT="${TRT_PORT:-8355}"

# Storage
HF_CACHE="${HF_CACHE:-/raid/hf-cache}"
TIKTOKEN_DIR="${TIKTOKEN_DIR:-${HOME}/tiktoken_encodings}"

# NCCL
NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-5}"
NCCL_TIMEOUT="${NCCL_TIMEOUT:-1200}"

# Worker configuration
# WORKER_HOST: Ethernet IP for SSH access (e.g., 192.168.7.111)
# WORKER_IB_IP: InfiniBand IP for NCCL/MPI communication (e.g., 169.254.216.8)
# Legacy WORKER_IPS is supported for backwards compatibility
WORKER_HOST="${WORKER_HOST:-}"
WORKER_IB_IP="${WORKER_IB_IP:-${WORKER_IPS:-}}"  # Fallback to WORKER_IPS for backwards compat
WORKER_USER="${WORKER_USER:-$(whoami)}"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Auto-detect Network Configuration (Head Node)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Auto-detect HEAD_IP from InfiniBand interface
if [ -z "${HEAD_IP:-}" ]; then
  if command -v ibdev2netdev >/dev/null 2>&1; then
    PRIMARY_IB_IF=$(ibdev2netdev 2>/dev/null | grep "(Up)" | awk '{print $5}' | grep "^enp1" | head -1)
    if [ -z "${PRIMARY_IB_IF}" ]; then
      PRIMARY_IB_IF=$(ibdev2netdev 2>/dev/null | grep "(Up)" | awk '{print $5}' | head -1)
    fi
    if [ -n "${PRIMARY_IB_IF}" ]; then
      HEAD_IP=$(ip -o addr show "${PRIMARY_IB_IF}" 2>/dev/null | awk '{print $4}' | cut -d'/' -f1 | head -1)
    fi
  fi
  if [ -z "${HEAD_IP:-}" ]; then
    echo "ERROR: Could not auto-detect HEAD_IP. Please set HEAD_IP in config.env"
    exit 1
  fi
fi

# Auto-detect network interfaces
if [ -z "${NCCL_SOCKET_IFNAME:-}" ] || [ -z "${GLOO_SOCKET_IFNAME:-}" ]; then
  if command -v ibdev2netdev >/dev/null 2>&1; then
    PRIMARY_IF=$(ibdev2netdev 2>/dev/null | grep "(Up)" | awk '{print $5}' | grep "^enp1" | head -1)
    if [ -z "${PRIMARY_IF}" ]; then
      PRIMARY_IF=$(ibdev2netdev 2>/dev/null | grep "(Up)" | awk '{print $5}' | head -1)
    fi
    NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${PRIMARY_IF}}"
    GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${PRIMARY_IF}}"
  fi
fi

# Auto-detect InfiniBand HCAs
if [ -z "${NCCL_IB_HCA:-}" ]; then
  if command -v ibdev2netdev >/dev/null 2>&1; then
    IB_DEVICES=$(ibdev2netdev 2>/dev/null | grep "(Up)" | awk '{print $1}' | sort | tr '\n' ',' | sed 's/,$//')
    NCCL_IB_HCA="${IB_DEVICES:-}"
  fi
  if [ -z "${NCCL_IB_HCA:-}" ]; then
    IB_DEVICES=$(ls -1 /sys/class/infiniband/ 2>/dev/null | tr '\n' ',' | sed 's/,$//')
    NCCL_IB_HCA="${IB_DEVICES:-}"
  fi
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

error() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
  exit 1
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Parse Arguments
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HEAD_ONLY=false
SKIP_PULL=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --head-only)
      HEAD_ONLY=true
      shift
      ;;
    --skip-pull)
      SKIP_PULL=true
      shift
      ;;
    --worker-ip|--worker-ib-ip)
      WORKER_IB_IP="$2"
      shift 2
      ;;
    --worker-host)
      WORKER_HOST="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --head-only          Only start head node (don't SSH to workers)"
      echo "  --skip-pull          Skip Docker image pull (faster restart)"
      echo "  --worker-host IP     Worker Ethernet IP for SSH (e.g., 192.168.7.111)"
      echo "  --worker-ib-ip IP    Worker InfiniBand IP for NCCL/MPI (e.g., 169.254.216.8)"
      echo "  -h, --help           Show this help"
      echo ""
      echo "Environment variables (recommended):"
      echo "  WORKER_HOST          Worker Ethernet IP for SSH"
      echo "  WORKER_IB_IP         Worker InfiniBand IP for NCCL/MPI"
      echo ""
      echo "Configuration is read from config.env or config.local.env"
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
# Validate Worker Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if [ "${HEAD_ONLY}" != "true" ] && [ "${NUM_NODES}" -gt 1 ]; then
  if [ -z "${WORKER_IB_IP}" ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " Worker Configuration Required"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "This is a ${NUM_NODES}-node cluster but no worker IPs are configured."
    echo ""
    echo "Please set these environment variables:"
    echo "  export WORKER_HOST=\"192.168.x.x\"    # Ethernet IP for SSH"
    echo "  export WORKER_IB_IP=\"169.254.x.x\"   # InfiniBand IP for NCCL/MPI"
    echo ""
    echo "Or start head only:"
    echo "  $0 --head-only"
    echo ""
    echo "To find worker IPs, run on the worker node:"
    echo "  hostname -I                          # Shows all IPs"
    echo "  ibdev2netdev && ip addr show <ib_if> # Shows IB interface IP"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    exit 1
  fi

  # If WORKER_HOST not set, fall back to WORKER_IB_IP for SSH (backwards compat)
  if [ -z "${WORKER_HOST}" ]; then
    log "Warning: WORKER_HOST not set, using WORKER_IB_IP (${WORKER_IB_IP}) for SSH"
    WORKER_HOST="${WORKER_IB_IP}"
  fi
fi

# Convert WORKER_IB_IP to array (supports multiple workers: "ip1 ip2 ip3")
read -ra WORKER_IB_IP_ARRAY <<< "${WORKER_IB_IP}"
read -ra WORKER_HOST_ARRAY <<< "${WORKER_HOST}"
ACTUAL_NUM_WORKERS=${#WORKER_IB_IP_ARRAY[@]}

if [ "${HEAD_ONLY}" != "true" ] && [ "${NUM_NODES}" -gt 1 ]; then
  EXPECTED_WORKERS=$((NUM_NODES - 1))
  if [ "${ACTUAL_NUM_WORKERS}" -ne "${EXPECTED_WORKERS}" ]; then
    log "Warning: NUM_NODES=${NUM_NODES} but only ${ACTUAL_NUM_WORKERS} worker IP(s) provided"
    log "Adjusting NUM_NODES to $((ACTUAL_NUM_WORKERS + 1))"
    NUM_NODES=$((ACTUAL_NUM_WORKERS + 1))
  fi
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Script
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " TensorRT-LLM DGX Spark Cluster Startup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
log "Configuration:"
log "  Model:             ${MODEL}"
log "  Tensor Parallel:   ${TENSOR_PARALLEL}"
log "  Nodes:             ${NUM_NODES}"
log "  Max Batch Size:    ${MAX_BATCH_SIZE}"
log "  Max Tokens:        ${MAX_NUM_TOKENS}"
log "  Backend:           ${TRT_BACKEND}"
log ""
log "Network:"
log "  Head IP:         ${HEAD_IP}"
log "  API Port:        ${TRT_PORT}"
if [ "${HEAD_ONLY}" != "true" ] && [ "${ACTUAL_NUM_WORKERS}" -gt 0 ]; then
  log "  Worker Host:     ${WORKER_HOST} (SSH)"
  log "  Worker IB IP:    ${WORKER_IB_IP} (NCCL/MPI)"
fi
log ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 1: Setup tiktoken encodings (for GPT-OSS models)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log "Step 1: Setting up tiktoken encodings"
mkdir -p "${TIKTOKEN_DIR}"

if [ ! -f "${TIKTOKEN_DIR}/o200k_base.tiktoken" ]; then
  log "  Downloading o200k_base.tiktoken..."
  wget -q -O "${TIKTOKEN_DIR}/o200k_base.tiktoken" \
    "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken" || \
    log "  Warning: Failed to download o200k_base.tiktoken"
fi

if [ ! -f "${TIKTOKEN_DIR}/cl100k_base.tiktoken" ]; then
  log "  Downloading cl100k_base.tiktoken..."
  wget -q -O "${TIKTOKEN_DIR}/cl100k_base.tiktoken" \
    "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken" || \
    log "  Warning: Failed to download cl100k_base.tiktoken"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 2: Pull Docker image
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if [ "${SKIP_PULL}" != "true" ]; then
  log "Step 2: Pulling Docker image on head node"
  docker pull "${TRT_IMAGE}" || error "Failed to pull image"
else
  log "Step 2: Skipping Docker pull (--skip-pull)"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 3: Clean up old head container
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log "Step 3: Cleaning up old containers"
if docker ps -a --format '{{.Names}}' | grep -qx "${HEAD_CONTAINER_NAME}"; then
  log "  Removing existing head container"
  docker rm -f "${HEAD_CONTAINER_NAME}" >/dev/null
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 4: Create TensorRT-LLM API config file
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log "Step 4: Creating TensorRT-LLM API config"

TRT_CONFIG_FILE="/tmp/trtllm-api-config.yml"
cat > "${TRT_CONFIG_FILE}" << EOF
print_iter_log: false
kv_cache_config:
  dtype: "auto"
  free_gpu_memory_fraction: ${GPU_MEMORY_FRACTION}
cuda_graph_config:
  enable_padding: ${CUDA_GRAPH_PADDING}
disable_overlap_scheduler: ${DISABLE_OVERLAP_SCHEDULER}
EOF

log "  Config written to ${TRT_CONFIG_FILE}"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 5: Start workers via SSH (before head, so they're ready to connect)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if [ "${HEAD_ONLY}" != "true" ] && [ "${ACTUAL_NUM_WORKERS}" -gt 0 ]; then
  log "Step 5: Starting workers via SSH"

  NODE_RANK=1
  for i in "${!WORKER_IB_IP_ARRAY[@]}"; do
    WORKER_IB="${WORKER_IB_IP_ARRAY[$i]}"
    # Use WORKER_HOST for SSH if available, otherwise fall back to IB IP
    SSH_HOST="${WORKER_HOST_ARRAY[$i]:-${WORKER_IB}}"
    log "  Starting worker at ${SSH_HOST} (IB: ${WORKER_IB}, node-rank ${NODE_RANK})..."

    # Test SSH connectivity
    if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "${WORKER_USER}@${SSH_HOST}" "echo ok" >/dev/null 2>&1; then
      error "Cannot SSH to ${WORKER_USER}@${SSH_HOST}. Check SSH keys and connectivity."
    fi

    # Start worker in background via SSH
    ssh "${WORKER_USER}@${SSH_HOST}" bash -s << WORKER_EOF &
set -e

# Configuration passed from head
export HEAD_IP="${HEAD_IP}"
export NODE_RANK="${NODE_RANK}"
export MODEL="${MODEL}"
export TENSOR_PARALLEL="${TENSOR_PARALLEL}"
export NUM_NODES="${NUM_NODES}"
export MAX_BATCH_SIZE="${MAX_BATCH_SIZE}"
export MAX_NUM_TOKENS="${MAX_NUM_TOKENS}"
export TRT_BACKEND="${TRT_BACKEND}"
export GPU_MEMORY_FRACTION="${GPU_MEMORY_FRACTION}"
export TRT_PORT="${TRT_PORT}"
export HF_CACHE="${HF_CACHE}"
export HF_TOKEN="${HF_TOKEN:-}"
export TRT_IMAGE="${TRT_IMAGE}"
export SHM_SIZE="${SHM_SIZE}"
export TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE}"
export EXTRA_ARGS="${EXTRA_ARGS}"
export NCCL_DEBUG="${NCCL_DEBUG}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT}"

# Setup tiktoken
TIKTOKEN_DIR="\${HOME}/tiktoken_encodings"
mkdir -p "\${TIKTOKEN_DIR}"
[ ! -f "\${TIKTOKEN_DIR}/o200k_base.tiktoken" ] && wget -q -O "\${TIKTOKEN_DIR}/o200k_base.tiktoken" "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken" || true
[ ! -f "\${TIKTOKEN_DIR}/cl100k_base.tiktoken" ] && wget -q -O "\${TIKTOKEN_DIR}/cl100k_base.tiktoken" "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken" || true

# Pull image if needed
docker pull "${TRT_IMAGE}" 2>/dev/null || true

# Auto-detect worker's own network settings
if command -v ibdev2netdev >/dev/null 2>&1; then
  PRIMARY_IF=\$(ibdev2netdev 2>/dev/null | grep "(Up)" | awk '{print \$5}' | grep "^enp1" | head -1)
  [ -z "\${PRIMARY_IF}" ] && PRIMARY_IF=\$(ibdev2netdev 2>/dev/null | grep "(Up)" | awk '{print \$5}' | head -1)
  NCCL_SOCKET_IFNAME="\${PRIMARY_IF}"
  GLOO_SOCKET_IFNAME="\${PRIMARY_IF}"
  IB_DEVICES=\$(ibdev2netdev 2>/dev/null | grep "(Up)" | awk '{print \$1}' | sort | tr '\n' ',' | sed 's/,\$//')
  NCCL_IB_HCA="\${IB_DEVICES}"
fi

# Clean up old container
WORKER_NAME="trtllm-worker-\$(hostname -s)"
docker rm -f "\${WORKER_NAME}" 2>/dev/null || true

# Create API config on worker
cat > /tmp/trtllm-api-config.yml << CONFIGEOF
print_iter_log: false
kv_cache_config:
  dtype: "auto"
  free_gpu_memory_fraction: ${GPU_MEMORY_FRACTION}
cuda_graph_config:
  enable_padding: true
disable_overlap_scheduler: true
CONFIGEOF

# Build environment args
ENV_ARGS="-e HF_TOKEN=\${HF_TOKEN:-} -e HF_HOME=/root/.cache/huggingface"
ENV_ARGS="\${ENV_ARGS} -e TIKTOKEN_ENCODINGS_BASE=/tiktoken_encodings"
ENV_ARGS="\${ENV_ARGS} -e NCCL_DEBUG=\${NCCL_DEBUG} -e NCCL_IB_DISABLE=\${NCCL_IB_DISABLE}"
ENV_ARGS="\${ENV_ARGS} -e NCCL_NET_GDR_LEVEL=\${NCCL_NET_GDR_LEVEL} -e NCCL_TIMEOUT=\${NCCL_TIMEOUT}"
[ -n "\${NCCL_SOCKET_IFNAME:-}" ] && ENV_ARGS="\${ENV_ARGS} -e NCCL_SOCKET_IFNAME=\${NCCL_SOCKET_IFNAME}"
[ -n "\${GLOO_SOCKET_IFNAME:-}" ] && ENV_ARGS="\${ENV_ARGS} -e GLOO_SOCKET_IFNAME=\${GLOO_SOCKET_IFNAME}"
[ -n "\${NCCL_IB_HCA:-}" ] && ENV_ARGS="\${ENV_ARGS} -e NCCL_IB_HCA=\${NCCL_IB_HCA}"

# Build TensorRT-LLM command
TRT_ARGS="--tp_size \${TENSOR_PARALLEL} --backend \${TRT_BACKEND}"
TRT_ARGS="\${TRT_ARGS} --max_num_tokens \${MAX_NUM_TOKENS} --max_batch_size \${MAX_BATCH_SIZE}"
TRT_ARGS="\${TRT_ARGS} --port \${TRT_PORT}"
[ "\${TRUST_REMOTE_CODE}" = "true" ] && TRT_ARGS="\${TRT_ARGS} --trust_remote_code"
[ -n "\${EXTRA_ARGS}" ] && TRT_ARGS="\${TRT_ARGS} \${EXTRA_ARGS}"

# Check for InfiniBand device
DEVICE_ARGS=""
[ -d "/dev/infiniband" ] && DEVICE_ARGS="--device=/dev/infiniband"

# Start container with MPI rank for multi-node
docker run -d \
  --restart no \
  --name "\${WORKER_NAME}" \
  --gpus all \
  --network host \
  --shm-size="\${SHM_SIZE}" \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ipc=host \
  \${DEVICE_ARGS} \
  -v "\${HF_CACHE}:/root/.cache/huggingface" \
  -v "\${TIKTOKEN_DIR}:/tiktoken_encodings" \
  -v "/tmp/trtllm-api-config.yml:/tmp/extra-llm-api-config.yml:ro" \
  \${ENV_ARGS} \
  "\${TRT_IMAGE}" \
  sleep infinity

echo "Worker \${WORKER_NAME} started on \$(hostname)"
WORKER_EOF

    NODE_RANK=$((NODE_RANK + 1))
  done

  # Wait briefly for workers to start
  log "  Waiting for workers to initialize..."
  sleep 5
else
  log "Step 5: Skipping workers (head-only mode or single node)"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 6: Start head node container
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log "Step 6: Starting head node container"

# Build environment variable arguments
ENV_ARGS=(
  -e "HF_TOKEN=${HF_TOKEN:-}"
  -e "HF_HOME=/root/.cache/huggingface"
  -e "TIKTOKEN_ENCODINGS_BASE=/tiktoken_encodings"
  -e "NCCL_DEBUG=${NCCL_DEBUG}"
  -e "NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
  -e "NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL}"
  -e "NCCL_TIMEOUT=${NCCL_TIMEOUT}"
)

[ -n "${NCCL_SOCKET_IFNAME:-}" ] && ENV_ARGS+=(-e "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}")
[ -n "${GLOO_SOCKET_IFNAME:-}" ] && ENV_ARGS+=(-e "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME}")
[ -n "${NCCL_IB_HCA:-}" ] && ENV_ARGS+=(-e "NCCL_IB_HCA=${NCCL_IB_HCA}")

# Volume mounts
VOLUME_ARGS=(
  -v "${HF_CACHE}:/root/.cache/huggingface"
  -v "${TIKTOKEN_DIR}:/tiktoken_encodings"
  -v "${TRT_CONFIG_FILE}:/tmp/extra-llm-api-config.yml:ro"
)

# Device args
DEVICE_ARGS=()
[ -d "/dev/infiniband" ] && DEVICE_ARGS+=(--device=/dev/infiniband)

# Start container in detached mode with sleep infinity
docker run -d \
  --restart no \
  --name "${HEAD_CONTAINER_NAME}" \
  --gpus all \
  --network host \
  --shm-size="${SHM_SIZE}" \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ipc=host \
  "${DEVICE_ARGS[@]}" \
  "${VOLUME_ARGS[@]}" \
  "${ENV_ARGS[@]}" \
  "${TRT_IMAGE}" \
  sleep infinity

if ! docker ps | grep -q "${HEAD_CONTAINER_NAME}"; then
  error "Head container failed to start. Check: docker logs ${HEAD_CONTAINER_NAME}"
fi

log "  Head container started"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 7: Build TensorRT-LLM serve command and start server
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log "Step 7: Starting TensorRT-LLM server"

# Build command arguments
TRT_CMD_ARGS="--tp_size ${TENSOR_PARALLEL}"
TRT_CMD_ARGS="${TRT_CMD_ARGS} --backend ${TRT_BACKEND}"
TRT_CMD_ARGS="${TRT_CMD_ARGS} --max_num_tokens ${MAX_NUM_TOKENS}"
TRT_CMD_ARGS="${TRT_CMD_ARGS} --max_batch_size ${MAX_BATCH_SIZE}"
TRT_CMD_ARGS="${TRT_CMD_ARGS} --port ${TRT_PORT}"

if [ "${TRUST_REMOTE_CODE}" = "true" ]; then
  TRT_CMD_ARGS="${TRT_CMD_ARGS} --trust_remote_code"
fi

if [ -n "${EXTRA_ARGS}" ]; then
  TRT_CMD_ARGS="${TRT_CMD_ARGS} ${EXTRA_ARGS}"
fi

# For multi-node, use MPI
if [ "${NUM_NODES}" -gt 1 ]; then
  # Create MPI hostfile (uses InfiniBand IPs for NCCL communication)
  HOSTFILE_CONTENT="${HEAD_IP} slots=1"
  for WORKER_IB in "${WORKER_IB_IP_ARRAY[@]}"; do
    HOSTFILE_CONTENT="${HOSTFILE_CONTENT}\n${WORKER_IB} slots=1"
  done

  docker exec "${HEAD_CONTAINER_NAME}" bash -c "echo -e '${HOSTFILE_CONTENT}' > /etc/openmpi-hostfile"

  log "  Starting multi-node TensorRT-LLM server with MPI..."
  docker exec -d "${HEAD_CONTAINER_NAME}" bash -c "
    mpirun --allow-run-as-root -x HF_TOKEN -x NCCL_DEBUG -x NCCL_IB_DISABLE -x NCCL_NET_GDR_LEVEL \
      --hostfile /etc/openmpi-hostfile \
      trtllm-llmapi-launch trtllm-serve '${MODEL}' ${TRT_CMD_ARGS} \
      > /var/log/trtllm.log 2>&1
  "
else
  log "  Starting single-node TensorRT-LLM server..."
  docker exec -d "${HEAD_CONTAINER_NAME}" bash -c "
    trtllm-serve '${MODEL}' ${TRT_CMD_ARGS} > /var/log/trtllm.log 2>&1
  "
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 8: Wait for cluster to be ready
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log "Step 8: Waiting for cluster to be ready"

if [ "${NUM_NODES}" -gt 1 ]; then
  MAX_WAIT=600
  log "  Multi-node cluster - waiting up to 10 minutes..."
else
  MAX_WAIT=300
  log "  Single-node - waiting up to 5 minutes..."
fi

READY=false
CONSECUTIVE_FAILURES=0
MAX_CONSECUTIVE_FAILURES=10

for i in $(seq 1 ${MAX_WAIT}); do
  if curl -sf "http://127.0.0.1:${TRT_PORT}/health" >/dev/null 2>&1; then
    log "  Cluster is ready! (${i}s)"
    READY=true
    break
  fi

  # Check if head container is still running
  CONTAINER_STATUS=$(docker inspect -f '{{.State.Status}}' "${HEAD_CONTAINER_NAME}" 2>/dev/null || echo "not_found")

  if [ "${CONTAINER_STATUS}" = "exited" ] || [ "${CONTAINER_STATUS}" = "dead" ]; then
    EXIT_CODE=$(docker inspect -f '{{.State.ExitCode}}' "${HEAD_CONTAINER_NAME}" 2>/dev/null || echo "unknown")
    if [ "${EXIT_CODE}" != "0" ]; then
      error "Head container exited with code ${EXIT_CODE}. Check: docker logs ${HEAD_CONTAINER_NAME}"
    fi
    CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
  elif [ "${CONTAINER_STATUS}" = "not_found" ]; then
    CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
  else
    CONSECUTIVE_FAILURES=0
  fi

  if [ ${CONSECUTIVE_FAILURES} -ge ${MAX_CONSECUTIVE_FAILURES} ]; then
    error "Head container not running after ${MAX_CONSECUTIVE_FAILURES} checks. Check: docker logs ${HEAD_CONTAINER_NAME}"
  fi

  # Progress every 30 seconds
  if [ $((i % 30)) -eq 0 ]; then
    log "  Still initializing... (${i}s)"
    docker exec "${HEAD_CONTAINER_NAME}" tail -2 /var/log/trtllm.log 2>&1 | grep -v "^$" | head -1 || true
  fi

  sleep 1
done

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Output Summary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Detect public-facing IP
PUBLIC_IP=$(ip -o addr show | grep "inet " | grep -v "127.0.0.1" | grep -v "169.254" | grep -v "172.17" | awk '{print $4}' | cut -d'/' -f1 | head -1)
[ -z "${PUBLIC_IP}" ] && PUBLIC_IP="${HEAD_IP}"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ "${READY}" = "true" ]; then
  echo " TensorRT-LLM Cluster is READY!"
else
  echo " TensorRT-LLM Cluster Started (still initializing)"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Cluster Info:"
echo "  Nodes:         ${NUM_NODES} (1 head + $((NUM_NODES - 1)) workers)"
echo "  Model:         ${MODEL}"
echo "  TP:            ${TENSOR_PARALLEL}"
echo "  Backend:       ${TRT_BACKEND}"
echo ""
echo "API Endpoints:"
echo "  API:           http://${PUBLIC_IP}:${TRT_PORT}/v1"
echo "  Health:        http://${PUBLIC_IP}:${TRT_PORT}/health"
echo ""
echo "Quick Test:"
echo "  curl http://${PUBLIC_IP}:${TRT_PORT}/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}'"
echo ""
echo "Logs:"
echo "  docker exec ${HEAD_CONTAINER_NAME} tail -f /var/log/trtllm.log"
for i in "${!WORKER_IB_IP_ARRAY[@]}"; do
  SSH_HOST="${WORKER_HOST_ARRAY[$i]:-${WORKER_IB_IP_ARRAY[$i]}}"
  echo "  ssh ${WORKER_USER}@${SSH_HOST} docker logs -f trtllm-worker-*"
done
echo ""
echo "Stop Cluster:"
echo "  ./stop_cluster.sh"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
