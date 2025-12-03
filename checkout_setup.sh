#!/usr/bin/env bash

################################################################################
# TensorRT-LLM Docker Swarm Checkout & Diagnostic Script
#
# Comprehensive diagnostic tool for multi-node GPU cluster setup.
# Checks head node, worker node(s), Docker Swarm, and GPU resource visibility.
#
# Usage:
#   ./checkout_setup.sh              # Interactive menu
#   ./checkout_setup.sh --quick      # Quick health check
#   ./checkout_setup.sh --full       # Full diagnostic (generates log file)
#   ./checkout_setup.sh --head       # Check head node only
#   ./checkout_setup.sh --worker     # Check worker node configuration
#   ./checkout_setup.sh --swarm      # Check Docker Swarm status
#   ./checkout_setup.sh --config     # Show current configuration
#
################################################################################

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load configuration
if [ -f "${SCRIPT_DIR}/config.local.env" ]; then
  source "${SCRIPT_DIR}/config.local.env"
elif [ -f "${SCRIPT_DIR}/config.env" ]; then
  source "${SCRIPT_DIR}/config.env"
fi

WORKER_HOST="${WORKER_HOST:-}"
WORKER_USER="${WORKER_USER:-$(whoami)}"

# Counters
ISSUES_FOUND=0
WARNINGS_FOUND=0

################################################################################
# Helper Functions
################################################################################

print_header() {
  echo ""
  echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "${BOLD}$1${NC}"
  echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_section() {
  echo ""
  echo -e "${CYAN}▶ $1${NC}"
}

print_ok() {
  echo -e "  ${GREEN}✓${NC} $1"
}

print_warn() {
  echo -e "  ${YELLOW}⚠${NC} $1"
  WARNINGS_FOUND=$((WARNINGS_FOUND + 1))
}

print_fail() {
  echo -e "  ${RED}✗${NC} $1"
  ISSUES_FOUND=$((ISSUES_FOUND + 1))
}

print_info() {
  echo -e "  ${BLUE}ℹ${NC} $1"
}

has_cmd() {
  command -v "$1" >/dev/null 2>&1
}

################################################################################
# Head Node Checks
################################################################################

check_head_node() {
  print_header "Head Node Diagnostics ($(hostname))"

  # 1. GPU Detection
  print_section "1. GPU Detection"
  if has_cmd nvidia-smi; then
    local gpu_count
    gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l || echo "0")
    if [ "$gpu_count" -gt 0 ]; then
      print_ok "$gpu_count GPU(s) detected"
      nvidia-smi -L 2>/dev/null | while read -r line; do
        echo "      $line"
      done

      # Get GPU UUID
      local gpu_uuid
      gpu_uuid=$(nvidia-smi -a 2>/dev/null | grep "GPU UUID" | head -1 | awk '{print $NF}')
      if [ -n "$gpu_uuid" ]; then
        print_ok "GPU UUID: $gpu_uuid"
      fi
    else
      print_fail "No GPUs detected"
    fi
  else
    print_fail "nvidia-smi not found"
  fi

  # 2. Docker Installation
  print_section "2. Docker Installation"
  if has_cmd docker; then
    print_ok "Docker installed: $(docker --version 2>/dev/null | head -1)"

    if docker info >/dev/null 2>&1; then
      print_ok "Docker daemon is running"
    else
      print_fail "Docker daemon is not running"
    fi
  else
    print_fail "Docker not installed"
  fi

  # 3. NVIDIA Container Runtime
  print_section "3. NVIDIA Container Runtime"
  if has_cmd nvidia-container-runtime; then
    print_ok "nvidia-container-runtime installed"
  else
    print_fail "nvidia-container-runtime not installed"
    print_info "Install: apt-get install -y nvidia-container-toolkit"
  fi

  # 4. Docker daemon.json
  print_section "4. Docker daemon.json Configuration"
  local daemon_json="/etc/docker/daemon.json"
  if [ -f "$daemon_json" ]; then
    print_ok "daemon.json exists"

    if grep -q "node-generic-resources" "$daemon_json" 2>/dev/null; then
      print_ok "node-generic-resources configured"
      local ngr
      ngr=$(grep -A1 "node-generic-resources" "$daemon_json" | tail -1 | tr -d '[]",' | xargs)
      echo "      $ngr"
    else
      print_fail "node-generic-resources NOT configured"
      print_info "This is required for Docker Swarm to see GPU resources"
    fi

    if grep -q '"nvidia"' "$daemon_json" 2>/dev/null; then
      print_ok "NVIDIA runtime configured"
    else
      print_warn "NVIDIA runtime may not be configured"
    fi

    echo ""
    echo "    Current daemon.json:"
    cat "$daemon_json" 2>/dev/null | sed 's/^/      /'
  else
    print_fail "daemon.json does not exist"
    print_info "Run ./setup_swarm.sh to configure"
  fi

  # 5. NVIDIA Container Runtime Config
  print_section "5. NVIDIA Container Runtime Config"
  local nvidia_config="/etc/nvidia-container-runtime/config.toml"
  if [ -f "$nvidia_config" ]; then
    print_ok "config.toml exists"

    if grep -q "^swarm-resource" "$nvidia_config" 2>/dev/null; then
      print_ok "swarm-resource enabled"
      grep "swarm-resource" "$nvidia_config" | sed 's/^/      /'
    else
      print_fail "swarm-resource NOT enabled (commented out or missing)"
      print_info "This is required for GPU visibility in Docker Swarm"
      grep -n "swarm-resource" "$nvidia_config" 2>/dev/null | sed 's/^/      /' || echo "      (line not found)"
    fi
  else
    print_fail "NVIDIA container runtime config not found"
    print_info "Expected at: $nvidia_config"
  fi

  # 6. Docker GPU Test
  print_section "6. Docker GPU Access Test"
  if docker info >/dev/null 2>&1; then
    local test_result
    test_result=$(docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu20.04 nvidia-smi -L 2>&1) || true
    if echo "$test_result" | grep -qi "gpu"; then
      print_ok "Docker can access GPUs"
      echo "$test_result" | head -2 | sed 's/^/      /'
    else
      print_fail "Docker cannot access GPUs"
      echo "$test_result" | head -3 | sed 's/^/      /'
    fi
  else
    print_warn "Cannot test GPU access - Docker not running"
  fi
}

################################################################################
# Worker Node Checks
################################################################################

check_worker_node() {
  print_header "Worker Node Diagnostics"

  if [ -z "$WORKER_HOST" ]; then
    print_warn "WORKER_HOST not configured"
    print_info "Set WORKER_HOST in config.local.env or environment"
    return 0
  fi

  local SSH_TARGET="${WORKER_USER}@${WORKER_HOST}"
  echo -e "  Target: ${CYAN}${SSH_TARGET}${NC}"
  echo ""

  # 1. SSH Connectivity
  print_section "1. SSH Connectivity"
  if ssh -o BatchMode=yes -o ConnectTimeout=5 "$SSH_TARGET" "echo ok" >/dev/null 2>&1; then
    print_ok "SSH connection successful"
  else
    print_fail "Cannot SSH to $SSH_TARGET"
    print_info "Check SSH keys: ssh-copy-id $SSH_TARGET"
    return 1
  fi

  # 2. Passwordless Sudo
  print_section "2. Passwordless Sudo"
  if ssh "$SSH_TARGET" "sudo -n true" 2>/dev/null; then
    print_ok "Passwordless sudo available"
  else
    print_fail "Passwordless sudo NOT available"
    print_info "On worker, run: sudo visudo"
    print_info "Add: ${WORKER_USER} ALL=(ALL) NOPASSWD: ALL"
    echo ""
    echo -e "    ${RED}This is likely the cause of your issue!${NC}"
    echo "    The setup script cannot configure the worker without passwordless sudo."
  fi

  # 3. GPU Detection
  print_section "3. GPU Detection"
  local gpu_info
  gpu_info=$(ssh "$SSH_TARGET" "nvidia-smi -L 2>/dev/null" 2>/dev/null) || true
  if [ -n "$gpu_info" ]; then
    local gpu_count
    gpu_count=$(echo "$gpu_info" | wc -l)
    print_ok "$gpu_count GPU(s) detected on worker"
    echo "$gpu_info" | sed 's/^/      /'

    local gpu_uuid
    gpu_uuid=$(ssh "$SSH_TARGET" "nvidia-smi -a 2>/dev/null | grep 'GPU UUID' | head -1 | awk '{print \$NF}'" 2>/dev/null) || true
    if [ -n "$gpu_uuid" ]; then
      print_ok "GPU UUID: $gpu_uuid"
    fi
  else
    print_fail "Cannot detect GPUs on worker"
  fi

  # 4. Docker Installation
  print_section "4. Docker Installation"
  if ssh "$SSH_TARGET" "docker --version" >/dev/null 2>&1; then
    local docker_ver
    docker_ver=$(ssh "$SSH_TARGET" "docker --version 2>/dev/null | head -1")
    print_ok "Docker installed: $docker_ver"

    if ssh "$SSH_TARGET" "docker info >/dev/null 2>&1"; then
      print_ok "Docker daemon is running"
    else
      print_fail "Docker daemon is not running"
    fi
  else
    print_fail "Docker not installed on worker"
  fi

  # 5. daemon.json
  print_section "5. Docker daemon.json Configuration"
  local daemon_exists
  daemon_exists=$(ssh "$SSH_TARGET" "test -f /etc/docker/daemon.json && echo yes || echo no" 2>/dev/null)

  if [ "$daemon_exists" = "yes" ]; then
    print_ok "daemon.json exists"

    if ssh "$SSH_TARGET" "grep -q 'node-generic-resources' /etc/docker/daemon.json" 2>/dev/null; then
      print_ok "node-generic-resources configured"
    else
      print_fail "node-generic-resources NOT configured"
    fi

    echo ""
    echo "    Current daemon.json on worker:"
    ssh "$SSH_TARGET" "cat /etc/docker/daemon.json 2>/dev/null" | sed 's/^/      /'
  else
    print_fail "daemon.json does NOT exist on worker"
    echo ""
    echo -e "    ${RED}This is likely the cause of your issue!${NC}"
    echo "    The setup script failed to create /etc/docker/daemon.json on the worker."
    echo "    Check passwordless sudo access above."
  fi

  # 6. NVIDIA Container Runtime Config
  print_section "6. NVIDIA Container Runtime Config"
  local nvidia_config_exists
  nvidia_config_exists=$(ssh "$SSH_TARGET" "test -f /etc/nvidia-container-runtime/config.toml && echo yes || echo no" 2>/dev/null)

  if [ "$nvidia_config_exists" = "yes" ]; then
    print_ok "config.toml exists"

    if ssh "$SSH_TARGET" "grep -q '^swarm-resource' /etc/nvidia-container-runtime/config.toml" 2>/dev/null; then
      print_ok "swarm-resource enabled"
    else
      print_fail "swarm-resource NOT enabled"
      echo "    Current swarm-resource line:"
      ssh "$SSH_TARGET" "grep 'swarm-resource' /etc/nvidia-container-runtime/config.toml 2>/dev/null" | sed 's/^/      /' || echo "      (not found)"
    fi
  else
    print_fail "NVIDIA container runtime config not found on worker"
  fi

  # 7. Swarm Membership
  print_section "7. Swarm Membership"
  local swarm_status
  swarm_status=$(ssh "$SSH_TARGET" "docker info 2>/dev/null | grep 'Swarm:' | awk '{print \$2}'" 2>/dev/null) || true

  if [ "$swarm_status" = "active" ]; then
    print_ok "Worker is in swarm (Swarm: active)"
  else
    print_info "Worker is not in swarm (Swarm: $swarm_status)"
  fi
}

################################################################################
# Docker Swarm Checks
################################################################################

check_swarm() {
  print_header "Docker Swarm Status"

  # 1. Swarm Status
  print_section "1. Swarm Mode Status"
  local swarm_status
  swarm_status=$(docker info 2>/dev/null | grep "Swarm:" | awk '{print $2}')

  if [ "$swarm_status" = "active" ]; then
    print_ok "Docker Swarm is active"
  else
    print_warn "Docker Swarm is not active (status: $swarm_status)"
    print_info "Run ./setup_swarm.sh to initialize"
    return 0
  fi

  # 2. Node List
  print_section "2. Swarm Nodes"
  echo ""
  docker node ls 2>/dev/null | sed 's/^/    /' || print_fail "Cannot list swarm nodes"

  # 3. GPU Resources
  print_section "3. GPU Resources in Swarm"
  local all_ok=true

  for node in $(docker node ls --format '{{.Hostname}}' 2>/dev/null); do
    local gpu_resources
    gpu_resources=$(docker node inspect "$node" --format '{{.Description.Resources.GenericResources}}' 2>/dev/null)

    if [ "$gpu_resources" = "[]" ] || [ -z "$gpu_resources" ]; then
      print_fail "Node $node: NO GPU resources visible"
      all_ok=false
    else
      print_ok "Node $node: GPU resources visible"
      echo "      $gpu_resources"
    fi
  done

  if [ "$all_ok" = "false" ]; then
    echo ""
    echo -e "  ${YELLOW}GPU resources not visible on some nodes.${NC}"
    echo "  This usually means:"
    echo "    1. daemon.json is missing node-generic-resources"
    echo "    2. swarm-resource is not enabled in NVIDIA config"
    echo "    3. Node needs to leave and rejoin swarm after Docker restart"
    echo ""
    echo "  Fix: Run ./setup_swarm.sh after fixing configuration"
  fi

  # 4. Services
  print_section "4. Docker Services"
  local services
  services=$(docker service ls 2>/dev/null)
  if [ -n "$services" ]; then
    echo "$services" | sed 's/^/    /'
  else
    print_info "No services running"
  fi

  # 5. Stacks
  print_section "5. Docker Stacks"
  local stacks
  stacks=$(docker stack ls 2>/dev/null)
  if [ -n "$stacks" ] && [ "$(echo "$stacks" | wc -l)" -gt 1 ]; then
    echo "$stacks" | sed 's/^/    /'
  else
    print_info "No stacks deployed"
  fi
}

################################################################################
# Quick Health Check
################################################################################

quick_check() {
  print_header "Quick Health Check"

  local all_ok=true

  # Head node GPU
  print_section "Head Node"
  local gpu_count
  gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l || echo "0")
  if [ "$gpu_count" -gt 0 ]; then
    print_ok "GPUs: $gpu_count detected"
  else
    print_fail "GPUs: None detected"
    all_ok=false
  fi

  # Docker
  if docker info >/dev/null 2>&1; then
    print_ok "Docker: Running"
  else
    print_fail "Docker: Not running"
    all_ok=false
  fi

  # daemon.json
  if [ -f /etc/docker/daemon.json ] && grep -q "node-generic-resources" /etc/docker/daemon.json 2>/dev/null; then
    print_ok "daemon.json: Configured"
  else
    print_fail "daemon.json: Not configured"
    all_ok=false
  fi

  # Swarm
  local swarm_status
  swarm_status=$(docker info 2>/dev/null | grep "Swarm:" | awk '{print $2}')
  if [ "$swarm_status" = "active" ]; then
    print_ok "Swarm: Active"
  else
    print_warn "Swarm: Not active"
  fi

  # Worker
  if [ -n "$WORKER_HOST" ]; then
    print_section "Worker Node ($WORKER_HOST)"

    if ssh -o BatchMode=yes -o ConnectTimeout=3 "${WORKER_USER}@${WORKER_HOST}" "echo ok" >/dev/null 2>&1; then
      print_ok "SSH: Connected"

      if ssh "${WORKER_USER}@${WORKER_HOST}" "sudo -n true" 2>/dev/null; then
        print_ok "Sudo: Passwordless OK"
      else
        print_fail "Sudo: Passwordless NOT available"
        all_ok=false
      fi

      if ssh "${WORKER_USER}@${WORKER_HOST}" "test -f /etc/docker/daemon.json && grep -q 'node-generic-resources' /etc/docker/daemon.json" 2>/dev/null; then
        print_ok "daemon.json: Configured"
      else
        print_fail "daemon.json: NOT configured"
        all_ok=false
      fi
    else
      print_fail "SSH: Cannot connect"
      all_ok=false
    fi
  else
    print_section "Worker Node"
    print_info "WORKER_HOST not configured"
  fi

  # GPU resources in swarm
  if [ "$swarm_status" = "active" ]; then
    print_section "Swarm GPU Resources"
    for node in $(docker node ls --format '{{.Hostname}}' 2>/dev/null); do
      local gpu_resources
      gpu_resources=$(docker node inspect "$node" --format '{{.Description.Resources.GenericResources}}' 2>/dev/null)
      if [ "$gpu_resources" = "[]" ] || [ -z "$gpu_resources" ]; then
        print_fail "$node: No GPU resources"
        all_ok=false
      else
        print_ok "$node: GPU resources visible"
      fi
    done
  fi

  echo ""
  if [ "$all_ok" = true ] && [ "$ISSUES_FOUND" -eq 0 ] && [ "$WARNINGS_FOUND" -eq 0 ]; then
    echo -e "${GREEN}All checks passed!${NC}"
  elif [ "$ISSUES_FOUND" -eq 0 ]; then
    echo -e "${YELLOW}$WARNINGS_FOUND warning(s) - review above${NC}"
  else
    echo -e "${RED}$ISSUES_FOUND issue(s) found - review above${NC}"
  fi
}

################################################################################
# Full Diagnostic
################################################################################

run_full_diagnostic() {
  local timestamp
  timestamp=$(date +"%Y%m%d_%H%M%S")
  local log_file="${SCRIPT_DIR}/diagnostic_report_${timestamp}.log"

  print_header "Full System Diagnostic"
  echo -e "  Generating comprehensive report..."
  echo -e "  Log file: ${GREEN}${log_file}${NC}"
  echo ""

  # Start log file
  {
    echo "TensorRT-LLM Docker Swarm Diagnostic Report"
    echo "============================================"
    echo "Generated: $(date)"
    echo "Hostname:  $(hostname)"
    echo "User:      $(whoami)"
    echo "Worker:    ${WORKER_HOST:-<not configured>}"
    echo ""
  } > "$log_file"

  # Helper to log commands
  log_cmd() {
    local desc="$1"
    local cmd="$2"
    echo ">>> $desc" | tee -a "$log_file"
    echo "Command: $cmd" >> "$log_file"
    echo "---" >> "$log_file"
    if eval "$cmd" >> "$log_file" 2>&1; then
      echo -e "  ${GREEN}✓${NC} $desc"
    else
      echo -e "  ${YELLOW}⚠${NC} $desc (may have failed)"
    fi
    echo "" >> "$log_file"
  }

  # 1. System Information
  print_section "1. System Information"
  log_cmd "OS Release" "cat /etc/os-release"
  log_cmd "Kernel" "uname -a"
  log_cmd "Hostname" "hostname"
  log_cmd "Memory" "free -h"

  # 2. GPU Information
  print_section "2. GPU Configuration"
  log_cmd "GPU List" "nvidia-smi -L"
  log_cmd "GPU Status" "nvidia-smi"
  log_cmd "GPU UUID" "nvidia-smi -a | grep 'GPU UUID'"

  # 3. Docker Configuration
  print_section "3. Docker Configuration"
  log_cmd "Docker Version" "docker --version"
  log_cmd "Docker Info" "docker info"
  log_cmd "daemon.json" "cat /etc/docker/daemon.json"
  log_cmd "NVIDIA Config" "cat /etc/nvidia-container-runtime/config.toml"

  # 4. Docker Swarm
  print_section "4. Docker Swarm"
  log_cmd "Swarm Status" "docker info | grep -A 10 'Swarm:'"
  log_cmd "Node List" "docker node ls"
  log_cmd "Node Inspect (all)" "for n in \$(docker node ls -q); do echo \"=== Node: \$n ===\"; docker node inspect \$n --format '{{.Description.Hostname}}: {{.Description.Resources.GenericResources}}'; done"

  # 5. Worker Node (if configured)
  if [ -n "$WORKER_HOST" ]; then
    print_section "5. Worker Node ($WORKER_HOST)"
    log_cmd "SSH Test" "ssh -o ConnectTimeout=5 ${WORKER_USER}@${WORKER_HOST} 'echo SSH OK'"
    log_cmd "Worker GPU" "ssh ${WORKER_USER}@${WORKER_HOST} 'nvidia-smi -L' 2>/dev/null || echo 'Failed'"
    log_cmd "Worker daemon.json" "ssh ${WORKER_USER}@${WORKER_HOST} 'cat /etc/docker/daemon.json' 2>/dev/null || echo 'Failed'"
    log_cmd "Worker NVIDIA Config" "ssh ${WORKER_USER}@${WORKER_HOST} 'grep swarm-resource /etc/nvidia-container-runtime/config.toml' 2>/dev/null || echo 'Failed'"
    log_cmd "Worker Docker Info" "ssh ${WORKER_USER}@${WORKER_HOST} 'docker info' 2>/dev/null || echo 'Failed'"
  fi

  # 6. Network
  print_section "6. Network Configuration"
  log_cmd "IP Addresses" "ip addr show"
  log_cmd "InfiniBand Devices" "ibdev2netdev 2>/dev/null || echo 'ibdev2netdev not available'"

  # 7. Running Containers
  print_section "7. Running Containers"
  log_cmd "Docker PS" "docker ps -a"
  log_cmd "Docker Services" "docker service ls 2>/dev/null || echo 'Not in swarm mode'"
  log_cmd "Docker Stacks" "docker stack ls 2>/dev/null || echo 'Not in swarm mode'"

  # Summary
  {
    echo ""
    echo "================================================================================"
    echo "  SUMMARY"
    echo "================================================================================"
    echo ""
    echo "Local GPUs:    $(nvidia-smi -L 2>/dev/null | wc -l || echo '0')"
    if [ -n "$WORKER_HOST" ]; then
      echo "Worker GPUs:   $(ssh ${WORKER_USER}@${WORKER_HOST} 'nvidia-smi -L 2>/dev/null | wc -l' 2>/dev/null || echo 'unknown')"
    fi
    echo "Swarm Active:  $(docker info 2>/dev/null | grep 'Swarm:' | awk '{print $2}')"
    echo "Swarm Nodes:   $(docker node ls 2>/dev/null | tail -n +2 | wc -l || echo '0')"
    echo ""
  } | tee -a "$log_file"

  echo ""
  echo -e "${GREEN}Diagnostic complete!${NC}"
  echo -e "Full report saved to: ${GREEN}${log_file}${NC}"
  echo ""
  echo "Share this log file when reporting issues."
}

################################################################################
# Show Configuration
################################################################################

show_config() {
  print_header "Current Configuration"

  print_section "Environment Variables"
  echo "  WORKER_HOST:   ${WORKER_HOST:-<not set>}"
  echo "  WORKER_USER:   ${WORKER_USER:-<not set>}"
  echo "  WORKER_IB_IP:  ${WORKER_IB_IP:-<not set>}"
  echo "  HEAD_IP:       ${HEAD_IP:-<not set>}"

  print_section "Config Files"
  if [ -f "${SCRIPT_DIR}/config.local.env" ]; then
    echo -e "  ${GREEN}config.local.env exists${NC}"
    echo ""
    cat "${SCRIPT_DIR}/config.local.env" | grep -v "^#" | grep -v "^$" | sed 's/^/    /'
  else
    echo "  config.local.env does not exist"
  fi

  print_section "Head Node Files"
  echo "  /etc/docker/daemon.json:"
  if [ -f /etc/docker/daemon.json ]; then
    cat /etc/docker/daemon.json | sed 's/^/    /'
  else
    echo "    (does not exist)"
  fi

  echo ""
  echo "  /etc/nvidia-container-runtime/config.toml (swarm-resource):"
  if [ -f /etc/nvidia-container-runtime/config.toml ]; then
    grep -n "swarm-resource" /etc/nvidia-container-runtime/config.toml | sed 's/^/    /'
  else
    echo "    (file does not exist)"
  fi

  if [ -n "$WORKER_HOST" ]; then
    print_section "Worker Node Files ($WORKER_HOST)"

    if ssh -o BatchMode=yes -o ConnectTimeout=3 "${WORKER_USER}@${WORKER_HOST}" "echo ok" >/dev/null 2>&1; then
      echo "  /etc/docker/daemon.json:"
      ssh "${WORKER_USER}@${WORKER_HOST}" "cat /etc/docker/daemon.json 2>/dev/null || echo '    (does not exist)'" | sed 's/^/    /'

      echo ""
      echo "  /etc/nvidia-container-runtime/config.toml (swarm-resource):"
      ssh "${WORKER_USER}@${WORKER_HOST}" "grep -n 'swarm-resource' /etc/nvidia-container-runtime/config.toml 2>/dev/null || echo '    (not found)'" | sed 's/^/    /'
    else
      echo "  Cannot SSH to worker"
    fi
  fi
}

################################################################################
# Interactive Menu
################################################################################

show_menu() {
  print_header "TensorRT-LLM Docker Swarm Checkout"
  echo ""
  echo "  Select an option:"
  echo ""
  echo "    1) Quick Health Check    - Fast status overview"
  echo "    2) Head Node Check       - Detailed head node diagnostics"
  echo "    3) Worker Node Check     - Detailed worker node diagnostics"
  echo "    4) Swarm Status          - Docker Swarm diagnostics"
  echo "    5) Full Diagnostic       - Complete report (generates log file)"
  echo "    6) Show Configuration    - Display current config"
  echo "    q) Quit"
  echo ""

  read -p "  Choice [1-6, q]: " choice

  case "$choice" in
    1) quick_check ;;
    2) check_head_node ;;
    3) check_worker_node ;;
    4) check_swarm ;;
    5) run_full_diagnostic ;;
    6) show_config ;;
    q|Q) echo "Exiting."; exit 0 ;;
    *) echo "Invalid choice"; exit 1 ;;
  esac
}

################################################################################
# Usage
################################################################################

usage() {
  cat << EOF
Usage: $0 [OPTION]

TensorRT-LLM Docker Swarm Checkout & Diagnostic Tool

Options:
  (no args)       Interactive menu
  --quick         Quick health check (recommended first step)
  --head          Detailed head node diagnostics
  --worker        Detailed worker node diagnostics
  --swarm         Docker Swarm status and GPU resources
  --full          Full diagnostic (generates log file for support)
  --config        Show current configuration
  -h, --help      Show this help

Environment Variables:
  WORKER_HOST     Worker node hostname/IP
  WORKER_USER     SSH username for worker (default: current user)

Examples:
  $0                            # Interactive menu
  $0 --quick                    # Quick health check
  $0 --worker                   # Check worker node setup
  $0 --full                     # Generate full diagnostic report

Common Issues:
  - "GPU resources not visible": Check daemon.json and swarm-resource config
  - "Cannot configure worker": Check passwordless sudo on worker
  - "Worker not in swarm": Run ./setup_swarm.sh

EOF
  exit 0
}

################################################################################
# Main
################################################################################

case "${1:-}" in
  --quick)
    quick_check
    ;;
  --head)
    check_head_node
    ;;
  --worker)
    check_worker_node
    ;;
  --swarm)
    check_swarm
    ;;
  --full)
    run_full_diagnostic
    ;;
  --config)
    show_config
    ;;
  -h|--help)
    usage
    ;;
  "")
    show_menu
    ;;
  *)
    echo "Unknown option: $1"
    usage
    ;;
esac

# Print summary if issues/warnings found
if [ "$ISSUES_FOUND" -gt 0 ] || [ "$WARNINGS_FOUND" -gt 0 ]; then
  echo ""
  print_header "Summary"
  if [ "$ISSUES_FOUND" -gt 0 ]; then
    echo -e "  ${RED}$ISSUES_FOUND critical issue(s) found${NC}"
  fi
  if [ "$WARNINGS_FOUND" -gt 0 ]; then
    echo -e "  ${YELLOW}$WARNINGS_FOUND warning(s) found${NC}"
  fi
  echo ""
  echo "  For detailed diagnostics, run: $0 --full"
  echo ""
fi

exit $ISSUES_FOUND
