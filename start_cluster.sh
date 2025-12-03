#!/usr/bin/env bash
set -euo pipefail

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TensorRT-LLM DGX Spark Cluster - Docker Swarm Stack Deployment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Starts TensorRT-LLM on both head and worker nodes using Docker Swarm.
# Based on NVIDIA's recommended multi-node deployment approach.
#
# Prerequisites:
#   1. Run ./setup_swarm.sh on all nodes (one-time setup)
#   2. Ensure HF_TOKEN is set for gated models
#
# Run this script on the HEAD NODE.
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
TRT_IMAGE="${TRT_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc4}"
STACK_NAME="${STACK_NAME:-trtllm-multinode}"

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
NCCL_TIMEOUT="${NCCL_TIMEOUT:-1200000}"  # 20 minutes in ms (large models need time to init)

# Worker configuration
WORKER_HOST="${WORKER_HOST:-}"
WORKER_IB_IP="${WORKER_IB_IP:-${WORKER_IPS:-}}"
WORKER_USER="${WORKER_USER:-$(whoami)}"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Auto-detect Network Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Auto-detect HEAD_IP from InfiniBand interface
# Priority: 1) config value, 2) InfiniBand interface IP, 3) first routable IP
if [ -z "${HEAD_IP:-}" ]; then
  if command -v ibdev2netdev >/dev/null 2>&1; then
    # Try DGX Spark naming convention first (enp1*), then any InfiniBand interface
    PRIMARY_IB_IF=$(ibdev2netdev 2>/dev/null | grep "(Up)" | awk '{print $5}' | grep "^enp1" | head -1)
    [ -z "${PRIMARY_IB_IF}" ] && PRIMARY_IB_IF=$(ibdev2netdev 2>/dev/null | grep "(Up)" | awk '{print $5}' | head -1)
    [ -n "${PRIMARY_IB_IF}" ] && HEAD_IP=$(ip -o addr show "${PRIMARY_IB_IF}" 2>/dev/null | awk '{print $4}' | cut -d'/' -f1 | head -1)
  fi
  # Fallback to first routable IP if InfiniBand detection failed
  if [ -z "${HEAD_IP:-}" ]; then
    HEAD_IP=$(hostname -I | awk '{print $1}')
  fi
  if [ -z "${HEAD_IP:-}" ]; then
    echo "ERROR: Could not auto-detect HEAD_IP. Please set HEAD_IP in config.env"
    exit 1
  fi
fi

# Auto-detect network interfaces
# For Docker Swarm overlay networking, containers use eth0/eth1 instead of host interfaces
if [ -z "${NCCL_SOCKET_IFNAME:-}" ]; then
  if [ "${NUM_NODES:-1}" -gt 1 ]; then
    # Multi-node: Use eth0 (overlay network interface inside containers)
    # eth0 = overlay network (10.0.x.x), eth1 = docker_gwbridge (172.19.x.x, local only)
    NCCL_SOCKET_IFNAME="eth0"
    GLOO_SOCKET_IFNAME="eth0"
  elif command -v ibdev2netdev >/dev/null 2>&1; then
    # Single-node with host networking: detect host interface
    PRIMARY_IF=$(ibdev2netdev 2>/dev/null | grep "(Up)" | awk '{print $5}' | grep "^enp1" | head -1)
    [ -z "${PRIMARY_IF}" ] && PRIMARY_IF=$(ibdev2netdev 2>/dev/null | grep "(Up)" | awk '{print $5}' | head -1)
    NCCL_SOCKET_IFNAME="${PRIMARY_IF}"
    GLOO_SOCKET_IFNAME="${PRIMARY_IF}"
  fi
fi

# Auto-detect InfiniBand HCAs
if [ -z "${NCCL_IB_HCA:-}" ]; then
  if command -v ibdev2netdev >/dev/null 2>&1; then
    NCCL_IB_HCA=$(ibdev2netdev 2>/dev/null | grep "(Up)" | awk '{print $1}' | sort | tr '\n' ',' | sed 's/,$//')
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

SKIP_PULL=false
HEAD_ONLY=false
SKIP_DOWNLOAD=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-pull)
      SKIP_PULL=true
      shift
      ;;
    --skip-download)
      SKIP_DOWNLOAD=true
      shift
      ;;
    --head-only)
      HEAD_ONLY=true
      NUM_NODES=1
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --skip-pull      Skip Docker image pull"
      echo "  --skip-download  Skip model download (if already cached)"
      echo "  --head-only      Run single-node on head only"
      echo "  -h, --help       Show this help"
      echo ""
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Convert worker IP lists to arrays
read -ra WORKER_HOST_ARRAY <<< "${WORKER_HOST}"
read -ra WORKER_IB_IP_ARRAY <<< "${WORKER_IB_IP}"

# If WORKER_HOST is empty but WORKER_IB_IP has values, use IB IPs for SSH
if [ ${#WORKER_HOST_ARRAY[@]} -eq 0 ] && [ ${#WORKER_IB_IP_ARRAY[@]} -gt 0 ]; then
  log "Warning: WORKER_HOST not set, using WORKER_IB_IP (${WORKER_IB_IP}) for SSH"
  read -ra WORKER_HOST_ARRAY <<< "${WORKER_IB_IP}"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Display Configuration
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
if [ ${#WORKER_HOST_ARRAY[@]} -gt 0 ]; then
  log "  Workers:         ${WORKER_HOST_ARRAY[*]}"
fi
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 1: Setup tiktoken encodings
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log "Step 1: Setting up tiktoken encodings"
mkdir -p "${TIKTOKEN_DIR}"
[ ! -f "${TIKTOKEN_DIR}/o200k_base.tiktoken" ] && wget -q -O "${TIKTOKEN_DIR}/o200k_base.tiktoken" "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken" 2>/dev/null || true
[ ! -f "${TIKTOKEN_DIR}/cl100k_base.tiktoken" ] && wget -q -O "${TIKTOKEN_DIR}/cl100k_base.tiktoken" "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken" 2>/dev/null || true

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 2: Verify Docker Swarm
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log "Step 2: Verifying Docker Swarm"

SWARM_STATUS=$(docker info 2>/dev/null | grep "Swarm:" | awk '{print $2}')

if [ "${SWARM_STATUS}" = "inactive" ]; then
  log "  Initializing Docker Swarm..."
  docker swarm init --advertise-addr "${HEAD_IP}" 2>/dev/null || true

  if [ "${HEAD_ONLY}" != "true" ] && [ ${#WORKER_HOST_ARRAY[@]} -gt 0 ]; then
    JOIN_TOKEN=$(docker swarm join-token worker -q)
    for i in "${!WORKER_HOST_ARRAY[@]}"; do
      SSH_HOST="${WORKER_HOST_ARRAY[$i]}"
      WORKER_IB="${WORKER_IB_IP_ARRAY[$i]:-${SSH_HOST}}"
      log "  Worker ${SSH_HOST} joining swarm..."
      ssh "${WORKER_USER}@${SSH_HOST}" "docker swarm leave --force 2>/dev/null || true"
      ssh "${WORKER_USER}@${SSH_HOST}" "docker swarm join --token '${JOIN_TOKEN}' --advertise-addr '${WORKER_IB}' '${HEAD_IP}:2377'"
    done
  fi
else
  log "  Docker Swarm is active"
fi

# Verify nodes
log "  Swarm nodes:"
docker node ls --format "  {{.Hostname}}: {{.Status}}" 2>/dev/null || true

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 3: Pull Docker image
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if [ "${SKIP_PULL}" = "true" ]; then
  log "Step 3: Skipping Docker pull (--skip-pull)"
else
  log "Step 3: Pulling Docker image on all nodes..."
  docker pull "${TRT_IMAGE}" &
  for SSH_HOST in "${WORKER_HOST_ARRAY[@]}"; do
    ssh "${WORKER_USER}@${SSH_HOST}" "docker pull '${TRT_IMAGE}'" &
  done
  wait
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 4: Create TensorRT-LLM API config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log "Step 4: Creating TensorRT-LLM API config"

TRT_CONFIG_FILE="/tmp/trtllm-api-config.yml"
cat > "${TRT_CONFIG_FILE}" << EOF
print_iter_log: false
kv_cache_config:
  dtype: "auto"
  free_gpu_memory_fraction: ${GPU_MEMORY_FRACTION}
cuda_graph_config:
  enable_padding: true
EOF

# Copy to workers
for SSH_HOST in "${WORKER_HOST_ARRAY[@]}"; do
  scp -q "${TRT_CONFIG_FILE}" "${WORKER_USER}@${SSH_HOST}:/tmp/trtllm-api-config.yml"
done

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 4b: Generate SSH keys for MPI communication
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if [ "${NUM_NODES}" -gt 1 ]; then
  log "Step 4b: Setting up SSH keys for MPI"
  SSH_DIR="/tmp/trtllm-ssh"
  rm -rf "${SSH_DIR}"
  mkdir -p "${SSH_DIR}"

  # Generate new key pair for container SSH
  ssh-keygen -t rsa -b 4096 -f "${SSH_DIR}/id_rsa" -N "" -q
  cp "${SSH_DIR}/id_rsa.pub" "${SSH_DIR}/authorized_keys"

  # Copy to all workers
  for SSH_HOST in "${WORKER_HOST_ARRAY[@]}"; do
    ssh "${WORKER_USER}@${SSH_HOST}" "rm -rf ${SSH_DIR}; mkdir -p ${SSH_DIR}"
    scp -q "${SSH_DIR}"/* "${WORKER_USER}@${SSH_HOST}:${SSH_DIR}/"
  done
  log "  SSH keys distributed to all nodes"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 5: Remove existing stack/containers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log "Step 5: Cleaning up existing deployment"
docker stack rm "${STACK_NAME}" 2>/dev/null || true
docker rm -f trtllm-head 2>/dev/null || true

# Wait for stack removal to complete (network removal is async)
for i in $(seq 1 30); do
  if ! docker network ls --format '{{.Name}}' | grep -q "^${STACK_NAME}"; then
    break
  fi
  sleep 1
done
sleep 2

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 6: Deploy based on mode (single-node or multi-node)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if [ "${NUM_NODES}" -gt 1 ]; then
  # =========================================================================
  # Multi-node deployment using Docker Stack (NVIDIA recommended approach)
  # =========================================================================

  log "Step 6: Deploying multi-node stack"

  # Export variables for docker-compose.yml
  export TRT_IMAGE HF_TOKEN HF_CACHE TIKTOKEN_DIR
  export NCCL_DEBUG NCCL_IB_DISABLE NCCL_NET_GDR_LEVEL NCCL_TIMEOUT
  export NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME NCCL_IB_HCA

  # Deploy the stack
  docker stack deploy --detach=true -c "${SCRIPT_DIR}/docker-compose.yml" "${STACK_NAME}"

  # Wait for containers to start
  log "  Waiting for stack containers to start..."
  for i in $(seq 1 60); do
    RUNNING=$(docker stack ps "${STACK_NAME}" --filter "desired-state=running" --format "{{.ID}}" 2>/dev/null | wc -l)
    if [ "${RUNNING}" -ge "${NUM_NODES}" ]; then
      log "  All ${NUM_NODES} containers running"
      break
    fi
    sleep 2
  done

  # Get container ID (on this node)
  sleep 5
  CONTAINER_ID=$(docker ps -q -f name="${STACK_NAME}")
  if [ -z "${CONTAINER_ID}" ]; then
    error "Could not find running container for stack ${STACK_NAME}"
  fi
  log "  Container: ${CONTAINER_ID}"

  # Generate MPI hostfile from container IPs in overlay network
  log "Step 7: Creating MPI hostfile"
  # Docker Swarm DNS doesn't properly resolve node hostnames to specific containers,
  # so we need to discover the actual overlay IPs from each node

  NETWORK_NAME="${STACK_NAME}_trtllm-net"
  > /tmp/openmpi-hostfile  # Clear the file

  # Get list of nodes running this service
  NODES=$(docker service ps "${STACK_NAME}_trtllm" --filter "desired-state=running" --format '{{.Node}}')

  # Build hostfile with container IPs from each node
  # Create a mapping of swarm node hostnames to their SSH IPs
  declare -A NODE_TO_SSH_IP
  NODE_TO_SSH_IP["$(hostname)"]="local"
  for i in "${!WORKER_HOST_ARRAY[@]}"; do
    # Get the swarm node name for this worker via SSH
    WORKER_SSH="${WORKER_HOST_ARRAY[$i]}"
    WORKER_NODE_NAME=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "${WORKER_USER}@${WORKER_SSH}" hostname 2>/dev/null || echo "")
    if [ -n "${WORKER_NODE_NAME}" ]; then
      NODE_TO_SSH_IP["${WORKER_NODE_NAME}"]="${WORKER_SSH}"
    fi
  done

  for node in ${NODES}; do
    if [ "${node}" = "$(hostname)" ]; then
      # Local node - get container ID and inspect it directly for its overlay IP
      LOCAL_CONTAINER=$(docker ps --filter "name=trtllm-multinode_trtllm" --format '{{.ID}}' | head -1)
      LOCAL_IP=$(docker inspect "${LOCAL_CONTAINER}" \
        --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' 2>/dev/null)
      echo "${LOCAL_IP}" >> /tmp/openmpi-hostfile
      log "    Local container (${node}): ${LOCAL_IP}"
    else
      # Remote node - SSH to get the container IP
      WORKER_SSH_IP="${NODE_TO_SSH_IP[${node}]:-}"
      if [ -z "${WORKER_SSH_IP}" ]; then
        # Fallback: try the first worker IP
        WORKER_SSH_IP="${WORKER_HOST_ARRAY[0]:-${WORKER_IB_IP_ARRAY[0]:-}}"
      fi
      if [ -n "${WORKER_SSH_IP}" ]; then
        # Get container ID on remote node, then get its overlay network IP
        REMOTE_IP=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "${WORKER_USER}@${WORKER_SSH_IP}" \
          "CONTAINER=\$(docker ps --filter 'name=trtllm-multinode_trtllm' --format '{{.ID}}' | head -1); \
           docker inspect \"\${CONTAINER}\" --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'" 2>/dev/null)
        if [ -n "${REMOTE_IP}" ]; then
          echo "${REMOTE_IP}" >> /tmp/openmpi-hostfile
          log "    Remote container (${node}): ${REMOTE_IP}"
        else
          log "  WARNING: Could not get IP for ${node}"
        fi
      else
        log "  WARNING: No SSH IP mapping for node ${node}"
      fi
    fi
  done

  # Sort to ensure consistent ordering across nodes
  sort -o /tmp/openmpi-hostfile /tmp/openmpi-hostfile

  # Copy hostfile to container first
  docker cp /tmp/openmpi-hostfile "${CONTAINER_ID}":/etc/openmpi-hostfile

  # Wait for SSH to be ready on ALL hosts in the hostfile
  log "  Waiting for SSH to be ready on all containers..."
  MAX_SSH_WAIT=120
  for host in $(cat /tmp/openmpi-hostfile); do
    log "    Checking SSH on ${host}..."
    for i in $(seq 1 ${MAX_SSH_WAIT}); do
      if docker exec "${CONTAINER_ID}" ssh -o ConnectTimeout=2 -o StrictHostKeyChecking=no -o BatchMode=yes "${host}" hostname >/dev/null 2>&1; then
        log "    SSH ready on ${host} (${i}s)"
        break
      fi
      if [ $((i % 10)) -eq 0 ]; then
        log "      Still waiting for SSH on ${host}... (${i}s)"
      fi
      sleep 1
    done
    if [ "${i}" -ge "${MAX_SSH_WAIT}" ]; then
      log "  WARNING: SSH not ready on ${host} after ${MAX_SSH_WAIT}s"
    fi
  done
  log "  Hostfile created with $(wc -l < /tmp/openmpi-hostfile) nodes:"
  cat /tmp/openmpi-hostfile

  # Download model (if not cached) - only on head node since cache is shared
  if [ "${SKIP_DOWNLOAD}" != "true" ]; then
    log "Step 8: Downloading model (head node only, shared cache)"
    # Build HF token arg if provided
    HF_TOKEN_ARG=""
    if [ -n "${HF_TOKEN:-}" ]; then
      HF_TOKEN_ARG="--token ${HF_TOKEN}"
    fi
    docker exec \
      -e HF_HOME=/root/.cache/huggingface \
      "${CONTAINER_ID}" bash -c "hf download ${MODEL} ${HF_TOKEN_ARG}" || true
  else
    log "Step 8: Skipping model download (--skip-download)"
  fi

  # Build serve command
  TRT_ARGS="--tp_size ${TENSOR_PARALLEL} --backend ${TRT_BACKEND}"
  TRT_ARGS="${TRT_ARGS} --max_num_tokens ${MAX_NUM_TOKENS} --max_batch_size ${MAX_BATCH_SIZE}"
  TRT_ARGS="${TRT_ARGS} --extra_llm_api_options /tmp/extra-llm-api-config.yml"
  TRT_ARGS="${TRT_ARGS} --host 0.0.0.0 --port ${TRT_PORT}"
  [ "${TRUST_REMOTE_CODE}" = "true" ] && TRT_ARGS="${TRT_ARGS} --trust_remote_code"
  [ -n "${EXTRA_ARGS}" ] && TRT_ARGS="${TRT_ARGS} ${EXTRA_ARGS}"

  # Start TensorRT-LLM server
  log "Step 9: Starting TensorRT-LLM server via MPI"

  # Create wrapper script that sets up environment for MPI remote processes
  docker exec "${CONTAINER_ID}" bash -c 'cat > /tmp/mpi-wrapper.sh << '\''WRAPPER'\''
#!/bin/bash
# Environment setup for MPI remote processes (Triton needs CUDA toolchain)
export PATH=/usr/local/cuda/bin:/usr/local/cmake/bin:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/bin:/usr/local/nvidia/bin:/usr/local/mpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/ucx/bin:/opt/amazon/efa/bin:/opt/tensorrt/bin
export CUDA_HOME=/usr/local/cuda
export CPATH=/usr/local/cuda/include
export C_INCLUDE_PATH=/usr/local/cuda/include
export CPLUS_INCLUDE_PATH=/usr/local/cuda/include
export LIBRARY_PATH=/usr/local/cuda/lib64
# Triton: Explicitly set CUDA tools paths
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TRITON_CUOBJDUMP_PATH=/usr/local/cuda/bin/cuobjdump
export TRITON_NVDISASM_PATH=/usr/local/cuda/bin/nvdisasm
# Full LD_LIBRARY_PATH from container (TensorRT, CUDA, PyTorch, etc.)
export LD_LIBRARY_PATH=/opt/nvidia/nvda_nixl/lib/aarch64-linux-gnu:/opt/nvidia/nvda_nixl/lib64:/usr/local/ucx/lib:/usr/local/tensorrt/lib:/usr/local/cuda/lib64:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
exec "$@"
WRAPPER
chmod +x /tmp/mpi-wrapper.sh'

  # Copy wrapper to remote nodes
  docker exec "${CONTAINER_ID}" bash -c 'for host in $(cat /etc/openmpi-hostfile); do scp -o StrictHostKeyChecking=no /tmp/mpi-wrapper.sh $host:/tmp/mpi-wrapper.sh 2>/dev/null || true; done'

  docker exec -d \
    -e MODEL="${MODEL}" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e TENSOR_PARALLEL="${TENSOR_PARALLEL}" \
    "${CONTAINER_ID}" bash -c "
      mpirun --allow-run-as-root -np ${TENSOR_PARALLEL} \
        --hostfile /etc/openmpi-hostfile \
        --map-by ppr:1:node \
        -x HF_TOKEN \
        /tmp/mpi-wrapper.sh trtllm-llmapi-launch trtllm-serve '${MODEL}' ${TRT_ARGS} \
        > /var/log/trtllm.log 2>&1
    "

else
  # =========================================================================
  # Single-node deployment (simple docker run)
  # =========================================================================

  log "Step 6: Starting single-node TensorRT-LLM server"

  TRT_ARGS="--tp_size 1 --backend ${TRT_BACKEND}"
  TRT_ARGS="${TRT_ARGS} --max_num_tokens ${MAX_NUM_TOKENS} --max_batch_size ${MAX_BATCH_SIZE}"
  TRT_ARGS="${TRT_ARGS} --port ${TRT_PORT}"
  [ "${TRUST_REMOTE_CODE}" = "true" ] && TRT_ARGS="${TRT_ARGS} --trust_remote_code"

  docker run -d \
    --name trtllm-head \
    --gpus all \
    --network host \
    --shm-size=32g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --ipc=host \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -v "${TIKTOKEN_DIR}:/tiktoken_encodings:ro" \
    -v "${TRT_CONFIG_FILE}:/tmp/extra-llm-api-config.yml:ro" \
    -e "HF_TOKEN=${HF_TOKEN:-}" \
    -e "HF_HOME=/root/.cache/huggingface" \
    -e "TIKTOKEN_ENCODINGS_BASE=/tiktoken_encodings" \
    "${TRT_IMAGE}" \
    trtllm-serve "${MODEL}" ${TRT_ARGS}

  CONTAINER_ID="trtllm-head"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Wait for cluster to be ready
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log "Step 10: Waiting for cluster to be ready"

if [ "${NUM_NODES}" -gt 1 ]; then
  MAX_WAIT=600
  log "  Multi-node cluster - waiting up to 10 minutes..."
else
  MAX_WAIT=300
  log "  Single-node - waiting up to 5 minutes..."
fi

READY=false

for i in $(seq 1 ${MAX_WAIT}); do
  if curl -sf "http://127.0.0.1:${TRT_PORT}/health" >/dev/null 2>&1; then
    log "  Cluster is ready! (${i}s)"
    READY=true
    break
  fi

  if [ $((i % 30)) -eq 0 ]; then
    log "  Still initializing... (${i}s)"
    # Show recent log output
    if [ "${NUM_NODES}" -gt 1 ]; then
      docker exec "${CONTAINER_ID}" tail -3 /var/log/trtllm.log 2>/dev/null | grep -v "^$" || true
    else
      docker logs --tail 3 "${CONTAINER_ID}" 2>&1 | grep -v "^$" || true
    fi
  fi

  sleep 1
done

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Summary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "${READY}" = "true" ]; then
  echo " TensorRT-LLM cluster is READY!"
  echo ""
  echo " Model:    ${MODEL}"
  echo " API:      http://127.0.0.1:${TRT_PORT}"
  echo " Nodes:    ${NUM_NODES}"
  echo ""
  echo " Test:"
  echo "   curl http://127.0.0.1:${TRT_PORT}/v1/models"
  echo ""
  echo " Chat:"
  echo "   curl -s http://localhost:${TRT_PORT}/v1/chat/completions \\"
  echo "     -H 'Content-Type: application/json' \\"
  echo "     -d '{\"model\": \"${MODEL}\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}'"
  echo ""
  echo " Logs:"
  if [ "${NUM_NODES}" -gt 1 ]; then
    echo "   docker exec ${CONTAINER_ID} tail -f /var/log/trtllm.log"
  else
    echo "   docker logs -f ${CONTAINER_ID}"
  fi
else
  echo " TensorRT-LLM cluster startup TIMED OUT"
  echo ""
  echo " Check logs:"
  if [ "${NUM_NODES}" -gt 1 ]; then
    echo "   docker exec ${CONTAINER_ID} cat /var/log/trtllm.log"
    echo "   docker stack ps ${STACK_NAME}"
  else
    echo "   docker logs ${CONTAINER_ID}"
  fi
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
