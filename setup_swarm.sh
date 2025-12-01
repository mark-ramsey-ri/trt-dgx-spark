#!/usr/bin/env bash
set -euo pipefail

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TensorRT-LLM Docker Swarm Setup Script
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# One-time setup for Docker Swarm with GPU resource advertising.
# Run this script on the HEAD NODE - it will configure workers via SSH.
# Requires sudo privileges on head and passwordless sudo on workers.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load configuration
if [ -f "${SCRIPT_DIR}/config.local.env" ]; then
  source "${SCRIPT_DIR}/config.local.env"
elif [ -f "${SCRIPT_DIR}/config.env" ]; then
  source "${SCRIPT_DIR}/config.env"
fi

# Worker configuration
WORKER_HOST="${WORKER_HOST:-}"
WORKER_IB_IP="${WORKER_IB_IP:-${WORKER_IPS:-}}"
WORKER_USER="${WORKER_USER:-$(whoami)}"

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

error() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
  exit 1
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Function to configure the local node
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

configure_local_node() {
  log "Configuring local node..."

  # Get GPU UUID
  local GPU_UUID
  GPU_UUID=$(nvidia-smi -a 2>/dev/null | grep "GPU UUID" | head -1 | awk '{print $NF}')

  if [ -z "${GPU_UUID}" ]; then
    error "Could not detect GPU UUID on local node"
  fi
  log "  GPU UUID: ${GPU_UUID}"

  local DAEMON_JSON="/etc/docker/daemon.json"
  local NVIDIA_CONFIG="/etc/nvidia-container-runtime/config.toml"

  echo "  Updating Docker daemon.json..."

  # Check if already configured
  if grep -q "node-generic-resources" "${DAEMON_JSON}" 2>/dev/null; then
    echo "    Docker daemon.json already configured"
  else
    # Backup existing config
    sudo cp "${DAEMON_JSON}" "${DAEMON_JSON}.backup.$(date +%Y%m%d%H%M%S)" 2>/dev/null || true

    # Create new daemon.json
    sudo tee "${DAEMON_JSON}" > /dev/null << EOF
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "nvidia",
  "node-generic-resources": [
    "NVIDIA_GPU=${GPU_UUID}"
  ]
}
EOF
    echo "    Updated daemon.json with GPU resource"
  fi

  echo "  Enabling swarm-resource in NVIDIA container runtime..."
  if grep -q "^swarm-resource" "${NVIDIA_CONFIG}" 2>/dev/null; then
    echo "    swarm-resource already enabled"
  else
    sudo sed -i 's/^#\s*\(swarm-resource\s*=\s*".*"\)/\1/' "${NVIDIA_CONFIG}"
    echo "    Enabled swarm-resource"
  fi

  echo "  Restarting Docker daemon..."
  sudo systemctl restart docker
  sleep 3

  if ! docker info >/dev/null 2>&1; then
    error "Docker failed to start on local node"
  fi
  echo "  Docker restarted successfully"

  log "  Local node configured successfully"
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Function to configure a remote node via SSH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

configure_remote_node() {
  local SSH_TARGET="$1"

  log "Configuring remote node (${SSH_TARGET})..."

  # Get GPU UUID from remote
  local GPU_UUID
  GPU_UUID=$(ssh "${SSH_TARGET}" "nvidia-smi -a 2>/dev/null | grep 'GPU UUID' | head -1 | awk '{print \$NF}'")

  if [ -z "${GPU_UUID}" ]; then
    error "Could not detect GPU UUID on remote node ${SSH_TARGET}"
  fi
  log "  GPU UUID: ${GPU_UUID}"

  # Build the remote script as a single command string
  local REMOTE_SCRIPT="
set -e
GPU_UUID='${GPU_UUID}'
DAEMON_JSON='/etc/docker/daemon.json'
NVIDIA_CONFIG='/etc/nvidia-container-runtime/config.toml'

echo '  Updating Docker daemon.json...'

if grep -q 'node-generic-resources' \"\${DAEMON_JSON}\" 2>/dev/null; then
  echo '    Docker daemon.json already configured'
else
  sudo cp \"\${DAEMON_JSON}\" \"\${DAEMON_JSON}.backup.\$(date +%Y%m%d%H%M%S)\" 2>/dev/null || true

  sudo tee \"\${DAEMON_JSON}\" > /dev/null << 'JSONEOF'
{
  \"runtimes\": {
    \"nvidia\": {
      \"path\": \"nvidia-container-runtime\",
      \"runtimeArgs\": []
    }
  },
  \"default-runtime\": \"nvidia\",
  \"node-generic-resources\": [
    \"NVIDIA_GPU=GPUPLACEHOLDER\"
  ]
}
JSONEOF
  sudo sed -i \"s/GPUPLACEHOLDER/\${GPU_UUID}/\" \"\${DAEMON_JSON}\"
  echo '    Updated daemon.json with GPU resource'
fi

echo '  Enabling swarm-resource in NVIDIA container runtime...'
if grep -q '^swarm-resource' \"\${NVIDIA_CONFIG}\" 2>/dev/null; then
  echo '    swarm-resource already enabled'
else
  sudo sed -i 's/^#[[:space:]]*\\(swarm-resource[[:space:]]*=[[:space:]]*\".*\"\\)/\\1/' \"\${NVIDIA_CONFIG}\"
  echo '    Enabled swarm-resource'
fi

echo '  Restarting Docker daemon...'
sudo systemctl restart docker
sleep 3

if ! docker info >/dev/null 2>&1; then
  echo 'ERROR: Docker failed to start'
  exit 1
fi
echo '  Docker restarted successfully'
"

  # Run with proper TTY allocation for sudo password prompt
  ssh -tt "${SSH_TARGET}" "bash -c '${REMOTE_SCRIPT}'"

  log "  Remote node configured successfully"
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Script
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " TensorRT-LLM Docker Swarm Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Convert worker IPs to array
read -ra WORKER_HOST_ARRAY <<< "${WORKER_HOST}"
read -ra WORKER_IB_IP_ARRAY <<< "${WORKER_IB_IP}"

# If WORKER_HOST is empty but WORKER_IB_IP has values, use IB IPs for SSH
if [ ${#WORKER_HOST_ARRAY[@]} -eq 0 ] && [ ${#WORKER_IB_IP_ARRAY[@]} -gt 0 ]; then
  read -ra WORKER_HOST_ARRAY <<< "${WORKER_IB_IP}"
fi

log "Will configure:"
echo "  - Head node (local)"
for host in "${WORKER_HOST_ARRAY[@]}"; do
  echo "  - Worker: ${host}"
done
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 1: Configure head node (local)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log "Step 1: Configuring head node"
configure_local_node

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 2: Configure worker nodes via SSH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if [ ${#WORKER_HOST_ARRAY[@]} -gt 0 ]; then
  STEP=2
  for host in "${WORKER_HOST_ARRAY[@]}"; do
    log "Step ${STEP}: Configuring worker ${host}"

    # Test SSH connectivity
    if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "${WORKER_USER}@${host}" "echo ok" >/dev/null 2>&1; then
      error "Cannot SSH to ${WORKER_USER}@${host}. Check SSH keys and connectivity."
    fi

    configure_remote_node "${WORKER_USER}@${host}"
    STEP=$((STEP + 1))
  done
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 3: Initialize or verify Docker Swarm
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log "Step 3: Setting up Docker Swarm"

# Check if already in swarm mode
if docker info 2>/dev/null | grep -q "Swarm: active"; then
  log "  Docker Swarm already active"
else
  log "  Initializing Docker Swarm on head node..."
  # Get the InfiniBand IP for swarm advertising (use first available)
  HEAD_ADVERTISE_IP=$(ip -4 addr show | grep -oP '169\.254\.\d+\.\d+' | head -1)
  if [ -z "${HEAD_ADVERTISE_IP}" ]; then
    # Fallback to any non-loopback IP
    HEAD_ADVERTISE_IP=$(hostname -I | awk '{print $1}')
  fi
  docker swarm init --advertise-addr "${HEAD_ADVERTISE_IP}" || true
fi

# Get join token for workers
JOIN_TOKEN=$(docker swarm join-token worker -q 2>/dev/null)
HEAD_ADVERTISE_IP=$(docker info 2>/dev/null | grep -A1 "Node Address" | tail -1 | tr -d ' ')
if [ -z "${HEAD_ADVERTISE_IP}" ]; then
  HEAD_ADVERTISE_IP=$(docker swarm join-token worker 2>/dev/null | grep -oP '\d+\.\d+\.\d+\.\d+:\d+' | head -1 | cut -d: -f1)
fi
SWARM_PORT=$(docker swarm join-token worker 2>/dev/null | grep -oP '\d+\.\d+\.\d+\.\d+:\d+' | head -1 | cut -d: -f2)
SWARM_PORT="${SWARM_PORT:-2377}"

# Join workers to swarm
if [ ${#WORKER_HOST_ARRAY[@]} -gt 0 ]; then
  for host in "${WORKER_HOST_ARRAY[@]}"; do
    log "  Joining worker ${host} to swarm..."

    # Check if worker is already in swarm
    WORKER_IN_SWARM=$(ssh "${WORKER_USER}@${host}" "docker info 2>/dev/null | grep -c 'Swarm: active'" || echo "0")

    if [ "${WORKER_IN_SWARM}" = "1" ]; then
      log "    Worker already in swarm"
    else
      # Leave any existing swarm first
      ssh "${WORKER_USER}@${host}" "docker swarm leave --force 2>/dev/null || true"
      # Join the swarm
      ssh "${WORKER_USER}@${host}" "docker swarm join --token ${JOIN_TOKEN} ${HEAD_ADVERTISE_IP}:${SWARM_PORT}"
      log "    Worker joined swarm"
    fi
  done
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 4: Verify GPU resources are visible in swarm
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log "Step 4: Verifying GPU resources in swarm"

# Wait a moment for swarm to sync
sleep 5

# Check each node for GPU resources
ALL_GPUS_OK=true
for node in $(docker node ls --format '{{.Hostname}}'); do
  GPU_RESOURCES=$(docker node inspect "${node}" --format '{{.Description.Resources.GenericResources}}' 2>/dev/null)
  if [ "${GPU_RESOURCES}" = "[]" ] || [ -z "${GPU_RESOURCES}" ]; then
    log "  WARNING: Node ${node} has no GPU resources visible!"
    log "    You may need to restart Docker on that node and re-run this script."
    ALL_GPUS_OK=false
  else
    log "  Node ${node}: GPU resources OK"
  fi
done

if [ "${ALL_GPUS_OK}" = "false" ]; then
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " WARNING: Some nodes don't have GPU resources visible!"
  echo ""
  echo " To fix, run on the affected node(s):"
  echo "   sudo systemctl restart docker"
  echo ""
  echo " Then re-run this script: ./setup_swarm.sh"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""
  exit 1
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Summary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Setup Complete!"
echo ""
echo " Docker Swarm configured with GPU resource advertising."
echo ""
echo " Swarm nodes:"
docker node ls --format "   - {{.Hostname}}: {{.Status}} ({{.ManagerStatus}})"
echo ""
echo " Next step:"
echo "   ./start_cluster.sh"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
