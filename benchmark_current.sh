#!/usr/bin/env bash
set -euo pipefail

################################################################################
# TensorRT-LLM Benchmark Script
#
# Benchmarks the currently running TensorRT-LLM server using OpenAI-compatible
# API endpoints with realistic workloads.
#
# Usage:
#   ./benchmark_current.sh [options]
#
# Options:
#   -u, --url URL           TensorRT-LLM API URL (default: auto-detect)
#   -n, --num-prompts N     Number of prompts to benchmark (default: 100)
#   -c, --concurrency N     Max concurrent requests (default: 32)
#   -d, --dataset PATH      Path to ShareGPT dataset JSON (auto-downloads if missing)
#   -s, --single            Run single-request benchmark only (1 prompt)
#   -q, --quick             Quick mode: 20 prompts, lower concurrency
#   -o, --output FILE       Output results to JSON file
#   -h, --help              Show this help message
#
# Examples:
#   ./benchmark_current.sh                    # Full benchmark (100 prompts)
#   ./benchmark_current.sh --quick            # Quick benchmark (20 prompts)
#   ./benchmark_current.sh --single           # Single request latency test
#   ./benchmark_current.sh -n 50 -c 16        # Custom prompts/concurrency
#
################################################################################

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load configuration
if [ -f "${SCRIPT_DIR}/config.local.env" ]; then
  source "${SCRIPT_DIR}/config.local.env"
elif [ -f "${SCRIPT_DIR}/config.env" ]; then
  source "${SCRIPT_DIR}/config.env"
fi

# Default configuration
TRT_PORT="${TRT_PORT:-8355}"
API_URL=""
NUM_PROMPTS=100
MAX_CONCURRENCY=32
DATASET_PATH=""
SINGLE_MODE=false
QUICK_MODE=false
OUTPUT_FILE=""
MODEL=""
CONTAINER="${HEAD_CONTAINER_NAME:-trtllm-head}"

# Default dataset location
DEFAULT_DATASET="ShareGPT_V3_unfiltered_cleaned_split.json"
DATASET_URL="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -u|--url)
      API_URL="$2"
      shift 2
      ;;
    -n|--num-prompts)
      NUM_PROMPTS="$2"
      shift 2
      ;;
    -c|--concurrency)
      MAX_CONCURRENCY="$2"
      shift 2
      ;;
    -d|--dataset)
      DATASET_PATH="$2"
      shift 2
      ;;
    -s|--single)
      SINGLE_MODE=true
      NUM_PROMPTS=1
      MAX_CONCURRENCY=1
      shift
      ;;
    -q|--quick)
      QUICK_MODE=true
      NUM_PROMPTS=20
      MAX_CONCURRENCY=16
      shift
      ;;
    -o|--output)
      OUTPUT_FILE="$2"
      shift 2
      ;;
    -h|--help)
      head -35 "$0" | tail -30
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

print_header() {
  echo ""
  echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "${BOLD}$1${NC}"
  echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_ok() {
  echo -e "${GREEN}✓${NC} $1"
}

print_warn() {
  echo -e "${YELLOW}⚠${NC} $1"
}

print_fail() {
  echo -e "${RED}✗${NC} $1"
}

print_info() {
  echo -e "${CYAN}ℹ${NC} $1"
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print_header "TensorRT-LLM Benchmark"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Auto-detect API URL if not provided
if [ -z "$API_URL" ]; then
  if curl -sf "http://localhost:${TRT_PORT}/health" >/dev/null 2>&1; then
    API_URL="http://localhost:${TRT_PORT}"
  else
    PUBLIC_IP=$(ip -o addr show | grep "inet " | grep -v "127.0.0.1" | grep -v "172.17" | awk '{print $4}' | cut -d'/' -f1 | head -1)
    if [ -n "$PUBLIC_IP" ] && curl -sf "http://${PUBLIC_IP}:${TRT_PORT}/health" >/dev/null 2>&1; then
      API_URL="http://${PUBLIC_IP}:${TRT_PORT}"
    else
      API_URL="http://localhost:${TRT_PORT}"
    fi
  fi
fi

echo ""
echo -e "  ${BOLD}Configuration:${NC}"
echo -e "    API URL:      ${CYAN}${API_URL}${NC}"
echo -e "    Num Prompts:  ${CYAN}${NUM_PROMPTS}${NC}"
echo -e "    Concurrency:  ${CYAN}${MAX_CONCURRENCY}${NC}"
if [ "$SINGLE_MODE" = true ]; then
  echo -e "    Mode:         ${CYAN}Single Request (latency test)${NC}"
elif [ "$QUICK_MODE" = true ]; then
  echo -e "    Mode:         ${CYAN}Quick${NC}"
else
  echo -e "    Mode:         ${CYAN}Full${NC}"
fi
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Check TensorRT-LLM availability
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo -e "${CYAN}▶${NC} Checking TensorRT-LLM availability..."

if ! curl -sf "${API_URL}/health" >/dev/null 2>&1; then
  print_fail "TensorRT-LLM is not accessible at ${API_URL}"
  echo "    Make sure TensorRT-LLM is running: ./start_cluster.sh"
  exit 1
fi
print_ok "TensorRT-LLM is accessible"

# Get model name from API
MODEL=$(curl -sf "${API_URL}/v1/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "unknown")
if [ "$MODEL" = "unknown" ]; then
  print_fail "Could not detect model from TensorRT-LLM API"
  exit 1
fi
print_ok "Model: ${MODEL}"
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Download ShareGPT dataset if needed
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo -e "${CYAN}▶${NC} Checking ShareGPT dataset..."

if [ -z "$DATASET_PATH" ]; then
  if [ -f "./${DEFAULT_DATASET}" ]; then
    DATASET_PATH="./${DEFAULT_DATASET}"
  elif [ -f "/tmp/${DEFAULT_DATASET}" ]; then
    DATASET_PATH="/tmp/${DEFAULT_DATASET}"
  else
    print_info "Downloading ShareGPT dataset..."
    if ! curl -sL "${DATASET_URL}" -o "/tmp/${DEFAULT_DATASET}"; then
      print_fail "Failed to download dataset"
      exit 1
    fi
    DATASET_PATH="/tmp/${DEFAULT_DATASET}"
  fi
fi

if [ ! -f "${DATASET_PATH}" ]; then
  print_fail "Dataset not found at ${DATASET_PATH}"
  exit 1
fi

DATASET_SIZE=$(wc -c < "${DATASET_PATH}" 2>/dev/null || echo "unknown")
print_ok "Dataset ready: ${DATASET_PATH} (${DATASET_SIZE} bytes)"
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run benchmark
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print_header "Running Benchmark"

echo ""
echo -e "  ${BOLD}Test Parameters:${NC}"
echo -e "    Model:        ${MODEL}"
echo -e "    Prompts:      ${NUM_PROMPTS}"
echo -e "    Concurrency:  ${MAX_CONCURRENCY}"
echo -e "    Dataset:      ShareGPT_V3"
echo ""

# Create Python benchmark script
BENCH_SCRIPT="/tmp/trt_benchmark_$$.py"

cat > "${BENCH_SCRIPT}" << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
"""
TensorRT-LLM Benchmark Script using OpenAI-compatible API
"""

import argparse
import json
import time
import random
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List
import urllib.request
import urllib.error

@dataclass
class BenchmarkResult:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    latencies: List[float] = field(default_factory=list)
    ttfts: List[float] = field(default_factory=list)
    start_time: float = 0
    end_time: float = 0

def load_sharegpt_prompts(dataset_path: str, num_prompts: int) -> List[str]:
    """Load prompts from ShareGPT dataset."""
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    prompts = []
    for item in data:
        conversations = item.get('conversations', [])
        for conv in conversations:
            if conv.get('from') == 'human':
                prompt = conv.get('value', '')
                if 50 < len(prompt) < 2000:  # Filter reasonable length prompts
                    prompts.append(prompt)
                    if len(prompts) >= num_prompts * 2:  # Get more than needed for random selection
                        break
        if len(prompts) >= num_prompts * 2:
            break

    # Random sample
    if len(prompts) > num_prompts:
        prompts = random.sample(prompts, num_prompts)

    return prompts[:num_prompts]

def make_request(api_url: str, model: str, prompt: str, max_tokens: int = 128) -> dict:
    """Make a single request to the API."""
    url = f"{api_url}/v1/chat/completions"

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }).encode('utf-8')

    headers = {
        "Content-Type": "application/json"
    }

    start_time = time.perf_counter()

    try:
        req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
        with urllib.request.urlopen(req, timeout=120) as response:
            response_data = json.loads(response.read().decode('utf-8'))

        end_time = time.perf_counter()
        latency = (end_time - start_time) * 1000  # Convert to ms

        usage = response_data.get('usage', {})
        output_tokens = usage.get('completion_tokens', 0)
        input_tokens = usage.get('prompt_tokens', 0)

        return {
            'success': True,
            'latency_ms': latency,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'ttft_ms': latency / max(output_tokens, 1) if output_tokens > 0 else latency  # Approximate TTFT
        }
    except Exception as e:
        end_time = time.perf_counter()
        return {
            'success': False,
            'latency_ms': (end_time - start_time) * 1000,
            'error': str(e),
            'input_tokens': 0,
            'output_tokens': 0,
            'ttft_ms': 0
        }

def run_benchmark(api_url: str, model: str, prompts: List[str], max_concurrency: int) -> BenchmarkResult:
    """Run the benchmark with concurrent requests."""
    result = BenchmarkResult()
    result.total_requests = len(prompts)
    result.start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {executor.submit(make_request, api_url, model, prompt): prompt for prompt in prompts}

        completed = 0
        for future in as_completed(futures):
            completed += 1
            resp = future.result()

            if resp['success']:
                result.successful_requests += 1
                result.latencies.append(resp['latency_ms'])
                result.ttfts.append(resp['ttft_ms'])
                result.total_input_tokens += resp['input_tokens']
                result.total_output_tokens += resp['output_tokens']
            else:
                result.failed_requests += 1

            if completed % 10 == 0:
                print(f"    Progress: {completed}/{len(prompts)} requests", flush=True)

    result.end_time = time.perf_counter()
    return result

def main():
    parser = argparse.ArgumentParser(description='TensorRT-LLM Benchmark')
    parser.add_argument('--api-url', required=True, help='API URL')
    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--dataset', required=True, help='Dataset path')
    parser.add_argument('--num-prompts', type=int, default=100, help='Number of prompts')
    parser.add_argument('--concurrency', type=int, default=32, help='Max concurrency')
    parser.add_argument('--output', help='Output JSON file')

    args = parser.parse_args()

    print(f"  Loading {args.num_prompts} prompts from dataset...")
    prompts = load_sharegpt_prompts(args.dataset, args.num_prompts)
    print(f"  Loaded {len(prompts)} prompts")
    print()

    print("  Starting benchmark...")
    result = run_benchmark(args.api_url, args.model, prompts, args.concurrency)

    # Calculate metrics
    duration = result.end_time - result.start_time

    if result.latencies:
        mean_latency = statistics.mean(result.latencies)
        p50_latency = statistics.median(result.latencies)
        p99_latency = sorted(result.latencies)[int(len(result.latencies) * 0.99)]
        mean_ttft = statistics.mean(result.ttfts)
    else:
        mean_latency = p50_latency = p99_latency = mean_ttft = 0

    output_tps = result.total_output_tokens / duration if duration > 0 else 0
    total_tps = (result.total_input_tokens + result.total_output_tokens) / duration if duration > 0 else 0
    req_throughput = result.successful_requests / duration if duration > 0 else 0

    # Print results in unified format
    print()
    print("  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  Benchmark Results")
    print("  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    print("  Test Configuration:")
    print(f"    Platform:           TensorRT-LLM")
    print(f"    Model:              {args.model}")
    print(f"    Num Prompts:        {args.num_prompts}")
    print(f"    Concurrency:        {args.concurrency}")
    print(f"    Dataset:            ShareGPT_V3")
    print()
    print("  Throughput Metrics:")
    print(f"    Duration:           {duration:.2f}s")
    print(f"    Requests/sec:       {req_throughput:.2f}")
    print(f"    Output tok/s:       {output_tps:.2f}")
    print(f"    Total tok/s:        {total_tps:.2f}")
    print()
    print("  Latency Metrics:")
    print(f"    Mean Latency:       {mean_latency:.2f} ms")
    print(f"    P50 Latency:        {p50_latency:.2f} ms")
    print(f"    P99 Latency:        {p99_latency:.2f} ms")
    print(f"    Mean TTFT:          {mean_ttft:.2f} ms")
    print()
    print("  Request Statistics:")
    print(f"    Completed:          {result.successful_requests}/{result.total_requests}")
    print(f"    Total Input Tokens: {result.total_input_tokens}")
    print(f"    Total Output Tokens:{result.total_output_tokens}")
    print()

    # Save to JSON if requested
    if args.output:
        output_data = {
            'platform': 'TensorRT-LLM',
            'model': args.model,
            'num_prompts': args.num_prompts,
            'concurrency': args.concurrency,
            'dataset': 'ShareGPT_V3',
            'duration_s': round(duration, 2),
            'successful_requests': result.successful_requests,
            'failed_requests': result.failed_requests,
            'output_throughput_tps': round(output_tps, 2),
            'total_throughput_tps': round(total_tps, 2),
            'request_throughput_rps': round(req_throughput, 2),
            'mean_latency_ms': round(mean_latency, 2),
            'p50_latency_ms': round(p50_latency, 2),
            'p99_latency_ms': round(p99_latency, 2),
            'mean_ttft_ms': round(mean_ttft, 2),
            'total_input_tokens': result.total_input_tokens,
            'total_output_tokens': result.total_output_tokens
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"  Results saved to: {args.output}")

if __name__ == '__main__':
    main()
PYTHON_SCRIPT

# Run the benchmark
OUTPUT_ARG=""
if [ -n "$OUTPUT_FILE" ]; then
  OUTPUT_ARG="--output ${OUTPUT_FILE}"
fi

python3 "${BENCH_SCRIPT}" \
  --api-url "${API_URL}" \
  --model "${MODEL}" \
  --dataset "${DATASET_PATH}" \
  --num-prompts "${NUM_PROMPTS}" \
  --concurrency "${MAX_CONCURRENCY}" \
  ${OUTPUT_ARG}

BENCH_EXIT_CODE=$?

# Cleanup
rm -f "${BENCH_SCRIPT}"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print_header "Benchmark Complete"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo ""
echo -e "  ${BOLD}Quick Reference Commands:${NC}"
echo ""
echo "  # Run single-request latency test"
echo "  ./benchmark_current.sh --single"
echo ""
echo "  # Run quick batch benchmark (20 prompts)"
echo "  ./benchmark_current.sh --quick"
echo ""
echo "  # Run full benchmark with JSON output"
echo "  ./benchmark_current.sh -n 100 -o results.json"
echo ""

exit $BENCH_EXIT_CODE
