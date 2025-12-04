#!/usr/bin/env bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TensorRT-LLM DGX Spark Environment Configuration Script
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# This script sets up environment variables for TensorRT-LLM cluster deployment.
# Network configuration (IPs, interfaces, HCAs) is auto-detected by scripts.
#
# Required configuration:
#   - WORKER_HOST: Worker Ethernet IP (for SSH access)
#   - WORKER_IB_IP: Worker InfiniBand IP (for NCCL communication)
#   - HF_TOKEN: HuggingFace token (for gated models like Llama)
#
# Usage:
#   source ./setup-env.sh           # Interactive mode (recommended)
#   source ./setup-env.sh --head    # Head node mode
#
# NOTE: This script must be SOURCED (not executed) to set environment variables
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Check if script is being sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Error: This script must be sourced, not executed"
    echo "Usage: source ./setup-env.sh"
    exit 1
fi

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Helper function to prompt for input
prompt_input() {
    local var_name="$1"
    local prompt_text="$2"
    local default_value="$3"
    local is_secret="${4:-false}"
    local current_value="${!var_name:-}"

    # If variable is already set, use it
    if [ -n "$current_value" ]; then
        if [ "$is_secret" = true ]; then
            echo -e "${GREEN}[ok]${NC} $var_name already set (hidden)"
        else
            echo -e "${GREEN}[ok]${NC} $var_name=$current_value"
        fi
        return
    fi

    # Show prompt
    if [ -n "$default_value" ]; then
        echo -ne "${BLUE}[?]${NC} $prompt_text [${default_value}]: "
    else
        echo -ne "${YELLOW}[!]${NC} $prompt_text: "
    fi

    # Read input (with or without echo for secrets)
    if [ "$is_secret" = true ]; then
        read -s user_input
        echo ""  # New line after secret input
    else
        read user_input
    fi

    # Use default if no input provided
    if [ -z "$user_input" ] && [ -n "$default_value" ]; then
        user_input="$default_value"
    fi

    # Export the variable
    if [ -n "$user_input" ]; then
        export "$var_name=$user_input"
        if [ "$is_secret" = true ]; then
            echo -e "${GREEN}[ok]${NC} $var_name set (hidden)"
        else
            echo -e "${GREEN}[ok]${NC} $var_name=$user_input"
        fi
    else
        if [ -n "$default_value" ]; then
            echo -e "${YELLOW}[-]${NC} $var_name not set (will use default: $default_value)"
        else
            echo -e "${YELLOW}[-]${NC} $var_name not set (optional)"
        fi
    fi
}

# Detect node type from arguments
NODE_TYPE="interactive"
if [[ "$1" == "--head" ]]; then
    NODE_TYPE="head"
fi

echo ""
echo -e "${GREEN}=============================================================${NC}"
echo -e "${GREEN}     TensorRT-LLM DGX Spark - Environment Setup${NC}"
echo -e "${GREEN}=============================================================${NC}"
echo ""
echo "Note: Network configuration (IPs, interfaces, HCAs) is auto-detected!"
echo "      You only need to provide the essential settings below."
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Check HuggingFace Cache Permissions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HF_CACHE="${HF_CACHE:-/raid/hf-cache}"
echo -e "${YELLOW}Checking HuggingFace cache...${NC}"

if [ -d "$HF_CACHE" ]; then
    if [ ! -w "$HF_CACHE" ]; then
        echo -e "${RED}[!]${NC} HF cache at $HF_CACHE is not writable"
        echo "    Docker containers run as root and may have created files owned by root."
        echo ""
        echo -e "${YELLOW}To fix, run:${NC}"
        echo "    sudo chown -R \$USER $HF_CACHE"
        echo ""
        read -p "Fix permissions now? (requires sudo) [y/N]: " fix_perms
        if [[ "$fix_perms" =~ ^[Yy]$ ]]; then
            if sudo chown -R "$USER" "$HF_CACHE"; then
                echo -e "${GREEN}[ok]${NC} Permissions fixed"
            else
                echo -e "${RED}[!]${NC} Failed. Please run manually: sudo chown -R \$USER $HF_CACHE"
            fi
        fi
    else
        echo -e "${GREEN}[ok]${NC} HF cache OK ($HF_CACHE)"
    fi
else
    echo -e "${BLUE}[i]${NC} HF cache will be created at $HF_CACHE"
fi
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Required Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo -e "${GREEN}--- Required Settings (CRITICAL for multi-node) ---${NC}"
echo ""

# Worker host for SSH
echo -e "${RED}WORKER_HOST - Worker IP address for SSH access:${NC}"
echo "  REQUIRED: This must be set for setup_swarm.sh to configure the worker!"
echo "  This is the IP address you use to SSH to the worker node."
echo ""
echo "  Find it on the worker by running: hostname -I | awk '{print \$1}'"
echo ""
echo "  Example: 192.168.7.111"
prompt_input "WORKER_HOST" "Enter worker IP address" ""
echo ""

# Worker high-speed network IP for NCCL
echo "WORKER_IB_IP - High-speed network IP (for NCCL/GPU communication):"
echo "  This is the IP on the InfiniBand/high-speed interface for RDMA"
echo "  Find it on worker: ibdev2netdev && ip addr show <ib-interface>"
echo "  Note: May be link-local (169.254.x.x) or routable depending on config"
prompt_input "WORKER_IB_IP" "Enter worker high-speed network IP" ""
echo ""

# Worker SSH username
echo "Worker Node Username (for SSH):"
prompt_input "WORKER_USER" "Enter worker username" "$(whoami)"
echo ""

# Head node IP for Docker Swarm
# Auto-detect first non-loopback IP as default
DEFAULT_HEAD_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo "HEAD_IP - Head node IP for Docker Swarm:"
echo "  Docker Swarm should use standard Ethernet IPs (not InfiniBand link-local)."
echo "  This ensures reliable swarm communication between nodes."
echo "  NCCL will still use InfiniBand for GPU transfers automatically."
echo ""
echo "  Find it on this node: hostname -I | awk '{print \$1}'"
prompt_input "HEAD_IP" "Enter head node Ethernet IP" "$DEFAULT_HEAD_IP"
echo ""

# HuggingFace Token
echo "HuggingFace Token (required for gated models like Llama):"
echo "  Get yours at: https://huggingface.co/settings/tokens"
echo "  Leave blank if using public models only"
prompt_input "HF_TOKEN" "Enter HuggingFace token" "" true
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Model Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo -e "${BLUE}--- Model Settings (press Enter for defaults) ---${NC}"
echo ""

echo "Model to serve:"
echo "  Recommended: nvidia/Qwen3-235B-A22B-FP4 (optimized for TRT-LLM)"
echo "  Alternatives: openai/gpt-oss-120b, meta-llama/Llama-3.3-70B-Instruct"
prompt_input "MODEL" "Model name" "nvidia/Qwen3-235B-A22B-FP4"
echo ""

prompt_input "TENSOR_PARALLEL" "Tensor parallel size (total GPUs)" "2"
prompt_input "NUM_NODES" "Number of nodes" "2"
prompt_input "MAX_BATCH_SIZE" "Maximum batch size" "4"
prompt_input "MAX_NUM_TOKENS" "Maximum tokens (context window)" "32768"
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Advanced Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo -e "${BLUE}--- Advanced Settings (press Enter for defaults) ---${NC}"
echo ""

prompt_input "TRT_IMAGE" "Docker image" "nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc4"
prompt_input "TRT_PORT" "API port" "8355"
prompt_input "GPU_MEMORY_FRACTION" "GPU memory fraction for KV cache (0.0-1.0)" "0.90"
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Summary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo -e "${GREEN}=============================================================${NC}"
echo -e "${GREEN}     Configuration Complete!${NC}"
echo -e "${GREEN}=============================================================${NC}"
echo ""
echo "Environment variables set:"
echo ""
[ -n "${WORKER_HOST:-}" ] && echo "  WORKER_HOST=$WORKER_HOST"
[ -n "${WORKER_IB_IP:-}" ] && echo "  WORKER_IB_IP=$WORKER_IB_IP"
[ -n "${WORKER_USER:-}" ] && echo "  WORKER_USER=$WORKER_USER"
[ -n "${HEAD_IP:-}" ] && echo "  HEAD_IP=$HEAD_IP"
[ -n "${HF_TOKEN:-}" ] && echo "  HF_TOKEN=(hidden)"
[ -n "${MODEL:-}" ] && echo "  MODEL=$MODEL"
[ -n "${TENSOR_PARALLEL:-}" ] && echo "  TENSOR_PARALLEL=$TENSOR_PARALLEL"
[ -n "${NUM_NODES:-}" ] && echo "  NUM_NODES=$NUM_NODES"
[ -n "${MAX_BATCH_SIZE:-}" ] && echo "  MAX_BATCH_SIZE=$MAX_BATCH_SIZE"
[ -n "${MAX_NUM_TOKENS:-}" ] && echo "  MAX_NUM_TOKENS=$MAX_NUM_TOKENS"
[ -n "${TRT_IMAGE:-}" ] && echo "  TRT_IMAGE=$TRT_IMAGE"
[ -n "${TRT_PORT:-}" ] && echo "  TRT_PORT=$TRT_PORT"
[ -n "${GPU_MEMORY_FRACTION:-}" ] && echo "  GPU_MEMORY_FRACTION=$GPU_MEMORY_FRACTION"
echo ""
echo "Auto-detected by scripts (no configuration needed):"
echo "  - Network interfaces (NCCL_SOCKET_IFNAME, GLOO_SOCKET_IFNAME)"
echo "  - InfiniBand HCAs (NCCL_IB_HCA)"
echo ""
if [ -z "${WORKER_HOST:-}" ]; then
  echo -e "${RED}WARNING: WORKER_HOST is not set!${NC}"
  echo "  Multi-node setup will NOT work without WORKER_HOST."
  echo "  Please set it before running setup_swarm.sh"
  echo ""
fi

echo -e "${GREEN}Next steps:${NC}"
echo ""
echo "  1. Save configuration (recommended):"
echo "     Add these settings to config.local.env"
echo ""
echo "  2. Run swarm setup (configures GPU for Docker Swarm):"
echo "     ./setup_swarm.sh"
echo ""
echo "  3. Start the cluster:"
echo "     ./start_cluster.sh"
echo ""
