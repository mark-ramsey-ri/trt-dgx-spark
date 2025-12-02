#!/usr/bin/env bash
set -euo pipefail

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TensorRT-LLM Model Switching Script
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Allows switching between different models with proper configuration.
# Handles tensor parallelism, node count, memory, and model-specific settings.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load configuration
if [ -f "${SCRIPT_DIR}/config.local.env" ]; then
  source "${SCRIPT_DIR}/config.local.env"
elif [ -f "${SCRIPT_DIR}/config.env" ]; then
  source "${SCRIPT_DIR}/config.env"
fi

HF_CACHE="${HF_CACHE:-/raid/hf-cache}"
HF_TOKEN="${HF_TOKEN:-}"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Model Definitions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Model HuggingFace IDs (matching vLLM/SGLang configuration)
MODELS=(
  "openai/gpt-oss-120b"
  "openai/gpt-oss-20b"
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen2.5-14B-Instruct"
  "Qwen/Qwen2.5-32B-Instruct"
  "Qwen/Qwen2.5-72B-Instruct"
  "mistralai/Mistral-7B-Instruct-v0.3"
  "mistralai/Mistral-Nemo-Instruct-2407"
  "mistralai/Mixtral-8x7B-Instruct-v0.1"
  "meta-llama/Llama-3.1-8B-Instruct"
  "meta-llama/Llama-3.1-70B-Instruct"
  "microsoft/phi-4"
  "google/gemma-2-27b-it"
)

# Human-readable model descriptions
MODEL_NAMES=(
  "GPT-OSS-120B (~80GB+, MoE reasoning model)"
  "GPT-OSS-20B (~16-20GB, MoE, fast)"
  "Qwen2.5-7B (~7GB, very fast)"
  "Qwen2.5-14B (~14GB, fast)"
  "Qwen2.5-32B (~30GB, strong mid-size)"
  "Qwen2.5-72B (~70GB, high quality)"
  "Mistral-7B v0.3 (~7GB, very fast)"
  "Mistral-Nemo-12B (~12GB, 128k context)"
  "Mixtral-8x7B (~45GB, MoE, fast)"
  "Llama-3.1-8B (~8GB, very fast)"
  "Llama-3.1-70B (~65GB, high quality)"
  "Phi-4 (~14-16GB, small but smart)"
  "Gemma2-27B (~24-28GB, strong mid-size)"
)

# Tensor Parallelism (number of GPUs needed)
# Always use TP=2 to run across both DGX Spark nodes
MODEL_TP=(
  2    # gpt-oss-120b
  2    # gpt-oss-20b
  2    # Qwen2.5-7B
  2    # Qwen2.5-14B
  2    # Qwen2.5-32B
  2    # Qwen2.5-72B
  2    # Mistral-7B
  2    # Mistral-Nemo-12B
  2    # Mixtral-8x7B
  2    # Llama-3.1-8B
  2    # Llama-3.1-70B
  2    # Phi-4
  2    # Gemma2-27B
)

# Number of nodes required (always 2 for DGX Spark cluster)
MODEL_NODES=(
  2    # gpt-oss-120b
  2    # gpt-oss-20b
  2    # Qwen2.5-7B
  2    # Qwen2.5-14B
  2    # Qwen2.5-32B
  2    # Qwen2.5-72B
  2    # Mistral-7B
  2    # Mistral-Nemo-12B
  2    # Mixtral-8x7B
  2    # Llama-3.1-8B
  2    # Llama-3.1-70B
  2    # Phi-4
  2    # Gemma2-27B
)

# GPU memory fraction (0.90 default, lower for larger models)
MODEL_MEM=(
  0.90  # gpt-oss-120b
  0.90  # gpt-oss-20b
  0.90  # Qwen2.5-7B
  0.90  # Qwen2.5-14B
  0.85  # Qwen2.5-32B
  0.90  # Qwen2.5-72B
  0.90  # Mistral-7B
  0.85  # Mistral-Nemo-12B (128k context)
  0.85  # Mixtral-8x7B (MoE)
  0.90  # Llama-3.1-8B
  0.90  # Llama-3.1-70B
  0.90  # Phi-4
  0.90  # Gemma2-27B
)

# Maximum number of tokens (context window)
MODEL_MAX_TOKENS=(
  8192    # gpt-oss-120b
  8192    # gpt-oss-20b
  32768   # Qwen2.5-7B
  32768   # Qwen2.5-14B
  32768   # Qwen2.5-32B
  32768   # Qwen2.5-72B
  32768   # Mistral-7B
  131072  # Mistral-Nemo-12B (128k context)
  32768   # Mixtral-8x7B
  131072  # Llama-3.1-8B (128k context)
  131072  # Llama-3.1-70B (128k context)
  16384   # Phi-4
  8192    # Gemma2-27B
)

# Maximum batch size
MODEL_BATCH_SIZE=(
  4     # gpt-oss-120b
  8     # gpt-oss-20b
  16    # Qwen2.5-7B
  16    # Qwen2.5-14B
  8     # Qwen2.5-32B
  4     # Qwen2.5-72B
  16    # Mistral-7B
  8     # Mistral-Nemo-12B
  8     # Mixtral-8x7B
  16    # Llama-3.1-8B
  4     # Llama-3.1-70B
  16    # Phi-4
  8     # Gemma2-27B
)

# Trust remote code flag
MODEL_TRUST_REMOTE=(
  false  # gpt-oss-120b
  false  # gpt-oss-20b
  false  # Qwen2.5-7B
  false  # Qwen2.5-14B
  false  # Qwen2.5-32B
  false  # Qwen2.5-72B
  false  # Mistral-7B
  false  # Mistral-Nemo-12B
  false  # Mixtral-8x7B
  false  # Llama-3.1-8B
  false  # Llama-3.1-70B
  true   # Phi-4 - requires trust_remote_code
  false  # Gemma2-27B
)

# Requires HF token (gated models)
MODEL_NEEDS_TOKEN=(
  false  # gpt-oss-120b
  false  # gpt-oss-20b
  false  # Qwen2.5-7B
  false  # Qwen2.5-14B
  false  # Qwen2.5-32B
  false  # Qwen2.5-72B
  false  # Mistral-7B
  false  # Mistral-Nemo-12B
  false  # Mixtral-8x7B
  true   # Llama-3.1-8B - gated
  true   # Llama-3.1-70B - gated
  false  # Phi-4
  true   # Gemma2-27B - gated
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

get_current_model() {
  if [ -f "${SCRIPT_DIR}/config.local.env" ]; then
    grep '^MODEL=' "${SCRIPT_DIR}/config.local.env" 2>/dev/null | head -1 | sed 's/MODEL="//' | sed 's/"$//' || echo ""
  elif [ -f "${SCRIPT_DIR}/config.env" ]; then
    grep '^MODEL=' "${SCRIPT_DIR}/config.env" 2>/dev/null | head -1 | sed 's/MODEL="\${MODEL:-//' | sed 's/}"$//' || echo ""
  else
    echo ""
  fi
}

check_hf_token() {
  if [ -n "${HF_TOKEN:-}" ]; then
    return 0
  fi
  if [ -f "${SCRIPT_DIR}/config.local.env" ]; then
    grep -q '^HF_TOKEN=' "${SCRIPT_DIR}/config.local.env" && return 0
  fi
  return 1
}

# Get the HF cache path for a model
get_model_cache_path() {
  local model="$1"
  local cache_name="models--$(echo "${model}" | sed 's|/|--|g')"
  echo "${HF_CACHE}/hub/${cache_name}"
}

# Check if model is downloaded locally
is_model_downloaded() {
  local model="$1"
  local cache_path=$(get_model_cache_path "${model}")

  if [ -d "${cache_path}/snapshots" ]; then
    local snapshot_count=$(find "${cache_path}/snapshots" -name "config.json" 2>/dev/null | wc -l)
    [ "${snapshot_count}" -gt 0 ]
  else
    return 1
  fi
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Parse Arguments
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SKIP_RESTART=false
LIST_ONLY=false
DOWNLOAD_ONLY=false
SKIP_DOWNLOAD=false
MODEL_NUMBER=""

usage() {
  cat << EOF
Usage: $0 [OPTIONS] [MODEL_NUMBER]

Switch between different models on the TensorRT-LLM cluster.

Options:
  -l, --list          List available models without switching
  -s, --skip-restart  Update config only, don't restart cluster
  -d, --download-only Download model only, don't switch or restart
  --skip-download     Skip download step (use existing cached model)
  -h, --help          Show this help

Examples:
  $0                  # Interactive model selection
  $0 1                # Switch to model #1 (Qwen3-235B-FP4)
  $0 --list           # List all available models
  $0 -s 3             # Update config for model #3 without restarting
  $0 -d 5             # Download model #5 only (no restart)

EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case $1 in
    -l|--list)
      LIST_ONLY=true
      shift
      ;;
    -s|--skip-restart)
      SKIP_RESTART=true
      shift
      ;;
    -d|--download-only)
      DOWNLOAD_ONLY=true
      shift
      ;;
    --skip-download)
      SKIP_DOWNLOAD=true
      shift
      ;;
    -h|--help)
      usage
      ;;
    [0-9]*)
      MODEL_NUMBER="$1"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Script
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " TensorRT-LLM Model Switcher"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Show current model
CURRENT_MODEL=$(get_current_model)
if [ -n "${CURRENT_MODEL}" ]; then
  echo "Current model: ${CURRENT_MODEL}"
else
  echo "Current model: (not configured)"
fi
echo ""

# Display available models
echo "Available models:"
echo ""
echo "  Multi-Node Models (TP=2, requires 2 DGX Spark nodes):"
for i in "${!MODELS[@]}"; do
  MARKER=""
  if [ "${MODELS[$i]}" = "${CURRENT_MODEL}" ]; then
    MARKER=" [CURRENT]"
  fi
  if [ "${MODEL_NEEDS_TOKEN[$i]}" = "true" ]; then
    MARKER="${MARKER} [HF TOKEN]"
  fi
  # Check if downloaded
  if is_model_downloaded "${MODELS[$i]}"; then
    MARKER="${MARKER} [CACHED]"
  fi
  printf "    %2d. %s%s\n" "$((i+1))" "${MODEL_NAMES[$i]}" "${MARKER}"
done
echo ""

# Exit if list only
if [ "${LIST_ONLY}" = "true" ]; then
  exit 0
fi

# Get model selection
if [ -z "${MODEL_NUMBER}" ]; then
  read -p "Select model (1-${#MODELS[@]}), or 'q' to quit: " MODEL_NUMBER
fi

if [ "${MODEL_NUMBER}" = "q" ] || [ "${MODEL_NUMBER}" = "Q" ]; then
  echo "Cancelled."
  exit 0
fi

# Validate selection
if ! [[ "${MODEL_NUMBER}" =~ ^[0-9]+$ ]] || [ "${MODEL_NUMBER}" -lt 1 ] || [ "${MODEL_NUMBER}" -gt "${#MODELS[@]}" ]; then
  echo "ERROR: Invalid selection. Please enter a number between 1 and ${#MODELS[@]}."
  exit 1
fi

# Get model configuration
IDX=$((MODEL_NUMBER - 1))
NEW_MODEL="${MODELS[$IDX]}"
NEW_MODEL_NAME="${MODEL_NAMES[$IDX]}"
NEW_TP="${MODEL_TP[$IDX]}"
NEW_NODES="${MODEL_NODES[$IDX]}"
NEW_MEM="${MODEL_MEM[$IDX]}"
NEW_MAX_TOKENS="${MODEL_MAX_TOKENS[$IDX]}"
NEW_BATCH_SIZE="${MODEL_BATCH_SIZE[$IDX]}"
NEW_TRUST="${MODEL_TRUST_REMOTE[$IDX]}"
NEEDS_TOKEN="${MODEL_NEEDS_TOKEN[$IDX]}"

# Check if model needs HF token
if [ "${NEEDS_TOKEN}" = "true" ]; then
  if ! check_hf_token; then
    echo ""
    echo "WARNING: ${NEW_MODEL} requires a HuggingFace token."
    echo ""
    echo "Please set HF_TOKEN before starting the cluster:"
    echo "  export HF_TOKEN=hf_your_token_here"
    echo ""
    echo "Or add to config.local.env:"
    echo "  HF_TOKEN=\"hf_your_token_here\""
    echo ""
    read -p "Continue anyway? (y/N): " CONTINUE
    if [ "${CONTINUE}" != "y" ] && [ "${CONTINUE}" != "Y" ]; then
      echo "Cancelled."
      exit 1
    fi
  fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Switching to: ${NEW_MODEL_NAME}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Configuration:"
echo "  Model:             ${NEW_MODEL}"
echo "  Tensor Parallel:   ${NEW_TP}"
echo "  Nodes Required:    ${NEW_NODES}"
echo "  GPU Memory Frac:   ${NEW_MEM}"
echo "  Max Tokens:        ${NEW_MAX_TOKENS}"
echo "  Max Batch Size:    ${NEW_BATCH_SIZE}"
[ "${NEW_TRUST}" = "true" ] && echo "  Trust Remote Code: yes"
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 1: Download Model (if needed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if [ "${SKIP_DOWNLOAD}" != "true" ]; then
  log "Step 1: Checking/Downloading model..."

  if is_model_downloaded "${NEW_MODEL}"; then
    log "  Model already cached locally"
  else
    log "  Model not found in cache, downloading..."
    TOKEN_ARG=""
    if [ -n "${HF_TOKEN:-}" ]; then
      TOKEN_ARG="--token ${HF_TOKEN}"
    fi
    if ! HF_HOME="${HF_CACHE}" huggingface-cli download "${NEW_MODEL}" ${TOKEN_ARG} --exclude "original/*" --exclude "metal/*" 2>&1 | tail -10; then
      echo "ERROR: Failed to download model"
      exit 1
    fi
    log "  Download complete"
  fi
else
  log "Skipping download (--skip-download specified)"
fi

# Exit if download only
if [ "${DOWNLOAD_ONLY}" = "true" ]; then
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " Download Complete"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""
  echo "Model downloaded: ${NEW_MODEL}"
  echo "Cache location:   $(get_model_cache_path "${NEW_MODEL}")"
  echo ""
  echo "To switch to this model and start the cluster:"
  echo "  $0 --skip-download ${MODEL_NUMBER}"
  echo ""
  exit 0
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 2: Update Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log ""
log "Step 2: Updating configuration..."

# Create or update config.local.env
CONFIG_FILE="${SCRIPT_DIR}/config.local.env"

# Start with existing config or empty
if [ -f "${CONFIG_FILE}" ]; then
  # Remove old model-related settings
  grep -v '^MODEL=\|^TENSOR_PARALLEL=\|^NUM_NODES=\|^GPU_MEMORY_FRACTION=\|^MAX_NUM_TOKENS=\|^MAX_BATCH_SIZE=\|^TRUST_REMOTE_CODE=\|^EXTRA_ARGS=' "${CONFIG_FILE}" > "${CONFIG_FILE}.tmp" || true
  mv "${CONFIG_FILE}.tmp" "${CONFIG_FILE}"
else
  # Copy from config.env template
  if [ -f "${SCRIPT_DIR}/config.env" ]; then
    cp "${SCRIPT_DIR}/config.env" "${CONFIG_FILE}"
    # Remove default values to override
    grep -v '^MODEL=\|^TENSOR_PARALLEL=\|^NUM_NODES=\|^GPU_MEMORY_FRACTION=\|^MAX_NUM_TOKENS=\|^MAX_BATCH_SIZE=\|^TRUST_REMOTE_CODE=\|^EXTRA_ARGS=' "${CONFIG_FILE}" > "${CONFIG_FILE}.tmp" || true
    mv "${CONFIG_FILE}.tmp" "${CONFIG_FILE}"
  else
    touch "${CONFIG_FILE}"
  fi
fi

# Add model configuration
{
  echo ""
  echo "# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "# Model Configuration (set by switch_model.sh)"
  echo "# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "MODEL=\"${NEW_MODEL}\""
  echo "TENSOR_PARALLEL=\"${NEW_TP}\""
  echo "NUM_NODES=\"${NEW_NODES}\""
  echo "GPU_MEMORY_FRACTION=\"${NEW_MEM}\""
  echo "MAX_NUM_TOKENS=\"${NEW_MAX_TOKENS}\""
  echo "MAX_BATCH_SIZE=\"${NEW_BATCH_SIZE}\""
  echo "TRUST_REMOTE_CODE=\"${NEW_TRUST}\""
  echo "EXTRA_ARGS=\"\""
} >> "${CONFIG_FILE}"

echo "  Configuration saved to: ${CONFIG_FILE}"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 3: Restart Cluster (if not skipped)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if [ "${SKIP_RESTART}" = "true" ]; then
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " Configuration Updated (restart skipped)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""
  echo "To start the cluster with the new model:"
  echo "  ./start_cluster.sh"
  echo ""
  exit 0
fi

# Stop existing cluster
echo ""
log "Step 3: Stopping existing cluster..."
if [ -x "${SCRIPT_DIR}/stop_cluster.sh" ]; then
  "${SCRIPT_DIR}/stop_cluster.sh" -f 2>/dev/null || true
else
  docker rm -f trtllm-head 2>/dev/null || true
fi
echo "  Cluster stopped"

# Start new cluster
echo ""
log "Step 4: Starting cluster with new model..."
echo ""

echo "  Starting cluster (this may take 3-5 minutes)..."
"${SCRIPT_DIR}/start_cluster.sh" --skip-pull 2>&1 | tee /tmp/model_switch.log &
STARTUP_PID=$!

# Wait for API
echo ""
log "Step 5: Waiting for API to become ready..."

MAX_WAIT=600
ELAPSED=0
TRT_PORT="${TRT_PORT:-8355}"
API_URL="http://127.0.0.1:${TRT_PORT}"

while [ $ELAPSED -lt $MAX_WAIT ]; do
  if curl -sf "${API_URL}/health" >/dev/null 2>&1; then
    echo ""
    echo "  API is ready!"
    break
  fi
  sleep 10
  ELAPSED=$((ELAPSED + 10))
  if [ $((ELAPSED % 30)) -eq 0 ]; then
    # Check if startup process is still running
    if ! kill -0 $STARTUP_PID 2>/dev/null; then
      # Check if it succeeded
      if curl -sf "${API_URL}/health" >/dev/null 2>&1; then
        echo ""
        echo "  API is ready!"
        break
      fi
    fi
    echo "  Still waiting... ${ELAPSED}s elapsed"
  fi
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
  echo ""
  echo "  WARNING: API not ready after ${MAX_WAIT}s"
  echo "  Check logs: docker logs trtllm-head"
  echo "  Or: cat /tmp/model_switch.log"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Verify and Display Results
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Model Switch Complete"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Get loaded model info
LOADED_MODEL=$(curl -sf "${API_URL}/v1/models" 2>/dev/null | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || echo "unknown")

echo "  Model:        ${LOADED_MODEL}"
echo "  API:          ${API_URL}"
echo "  Health:       ${API_URL}/health"
echo "  Time:         ${ELAPSED}s"
echo ""

# Quick test
echo "Testing inference..."
TEST_RESPONSE=$(curl -sf "${API_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"'"${NEW_MODEL}"'","messages":[{"role":"user","content":"Say OK"}],"max_tokens":5}' 2>/dev/null || echo "{}")

if echo "${TEST_RESPONSE}" | grep -q '"choices"'; then
  echo "  Inference test: PASSED"
else
  echo "  Inference test: FAILED (check logs)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
