# TensorRT-LLM on DGX Spark Cluster

Deploy [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) on a dual-node NVIDIA DGX Spark cluster with InfiniBand RDMA for serving large language models.

> **DISCLAIMER**: This project is NOT affiliated with, endorsed by, or officially supported by NVIDIA or any other organization. This is a community-driven effort to run TensorRT-LLM on DGX Spark hardware. Use at your own risk. The software is provided "AS IS", without warranty of any kind.

## Features

- **Single-command deployment** - Start entire cluster from head node via SSH
- **Docker Swarm integration** - GPU resource advertising for multi-node scheduling
- **Auto-detection** of InfiniBand IPs, network interfaces, and HCA devices
- **Generic scripts** that work on any DGX Spark pair
- **Multiple model presets** including Qwen3, Llama, and GPT-OSS
- **InfiniBand RDMA** for high-speed inter-node communication (200Gb/s)
- **Comprehensive benchmarking** with multiple test profiles

## Cluster Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DGX Spark 2-Node Cluster                     │
│                                                                 │
│  ┌──────────────────────┐      ┌──────────────────────┐        │
│  │     HEAD NODE        │      │    WORKER NODE       │        │
│  │  (TensorRT-LLM head) │ SSH  │  (TensorRT-LLM wkr)  │        │
│  │                      │─────►│                      │        │
│  │  GPU: 1x GB10        │◄────►│  GPU: 1x GB10        │        │
│  │  (Blackwell, sm100)  │ IB   │  (Blackwell, sm100)  │        │
│  │                      │200Gb │                      │        │
│  │  /raid/hf-cache      │      │  /raid/hf-cache      │        │
│  │  Port: 8355 (API)    │      │                      │        │
│  └──────────────────────┘      └──────────────────────┘        │
│                                                                 │
│  Tensor Parallel (TP=2): Model split across both GPUs          │
│  Docker Swarm: GPU resource advertising for scheduling         │
└─────────────────────────────────────────────────────────────────┘
```

## Hardware Requirements

- **Nodes:** 2x DGX Spark systems
- **GPUs:** 1x NVIDIA GB10 (Grace Blackwell, sm100) per node, ~120GB VRAM each
- **Network:** 200Gb/s InfiniBand RoCE between nodes
- **Storage:** Shared model cache at `/raid/hf-cache` (or configure in `config.env`)
- **SSH:** Passwordless SSH from head to worker node(s)
- **Sudo:** Passwordless sudo on **BOTH** head and worker nodes (required for Docker/swarm setup)

## Prerequisites

Complete these steps on **BOTH** servers before running `start_cluster.sh`:

### 1. NVIDIA GPU Drivers

Ensure NVIDIA drivers are installed and working:
```bash
nvidia-smi
```
You should see your GPU listed with driver version.

### 2. Docker with NVIDIA Container Runtime

Docker must be installed with NVIDIA Container Runtime configured:
```bash
# Verify Docker works with GPU access
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu20.04 nvidia-smi
```
If this fails, install/configure the NVIDIA Container Toolkit.

### 3. InfiniBand Network Configuration

**CRITICAL:** InfiniBand (QSFP) interfaces must be configured and operational for multi-node performance.

```bash
# Check InfiniBand status
ibstatus

# Find InfiniBand interfaces (typically enp1s0f1np1, enP2p1s0f1np1 on DGX Spark)
ip addr show | grep 169.254

# Verify both nodes can reach each other via InfiniBand
ping <infiniband-ip-of-other-node>
```

InfiniBand IPs are typically in the `169.254.x.x` range.

**Performance Warning:** Using standard Ethernet IPs instead of InfiniBand will result in **10-20x slower performance**.

Need help with InfiniBand setup? See NVIDIA's guide: https://build.nvidia.com/spark/nccl/stacked-sparks

### 4. Firewall Configuration

Ensure the following ports are open between both nodes:
- **2377** - Docker Swarm management
- **7946** - Docker Swarm node communication (TCP/UDP)
- **4789** - Docker Swarm overlay network (UDP)
- **8355** - TensorRT-LLM API

### 5. Passwordless Sudo (BOTH nodes)

The setup scripts need to modify Docker configuration and restart services on both nodes. Configure passwordless sudo on **BOTH** head and worker nodes:

```bash
# Run on EACH node (head AND worker):
echo "$USER ALL=(ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/$USER
sudo chmod 440 /etc/sudoers.d/$USER

# Verify it works:
sudo -n echo "SUDO_OK"
```

> **Why is this needed?** The `setup_swarm.sh` script must:
> - Modify `/etc/docker/daemon.json` to enable GPU resource advertising
> - Update `/etc/nvidia-container-runtime/config.toml` for swarm support
> - Restart the Docker daemon on both nodes

### 6. Hugging Face Authentication (for gated models)

Some models (Llama, Gemma, etc.) require Hugging Face authorization:

```bash
# Install the Hugging Face CLI (run on both nodes)
pip install huggingface_hub

# Login to Hugging Face (run on both nodes)
huggingface-cli login
# Enter your token when prompted

# Accept model licenses
# Visit the model page on huggingface.co and accept the license agreement
# Example: https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
```

Alternatively, set `HF_TOKEN` in your `config.local.env`:
```bash
HF_TOKEN="hf_your_token_here"
```

## Quick Start

### 1. Clone and Setup

```bash
git clone <this-repo>
cd trt-dgx-spark
```

### 2. Setup SSH (one-time)

Ensure passwordless SSH from head to worker using the **standard Ethernet IP**:
```bash
# On head node, generate key if needed:
ssh-keygen -t ed25519  # Press enter for defaults

# Copy to worker (use standard Ethernet IP, e.g., 192.168.x.x):
ssh-copy-id <username>@<worker-ethernet-ip>

# Test connection:
ssh <username>@<worker-ethernet-ip> "hostname"
```

> **Note:** Use the standard Ethernet IP for SSH (WORKER_HOST), not the InfiniBand IP.

### 3. Configure Environment

**Option A: Interactive setup (recommended)**
```bash
source ./setup-env.sh
```

**Option B: Edit config file**
```bash
cp config.env config.local.env
vim config.local.env

# Set at minimum:
# WORKER_HOST="<worker-ethernet-ip>"     # For SSH (e.g., 192.168.7.111)
# WORKER_IB_IP="<worker-infiniband-ip>"  # For NCCL (e.g., 169.254.216.8)
# WORKER_USER="<ssh-username>"
```

### 4. Setup Docker Swarm (one-time)

TensorRT-LLM uses Docker Swarm for multi-node GPU scheduling:
```bash
./setup_swarm.sh
```

This script will:
1. Configure GPU resource advertising in Docker daemon
2. Enable swarm-resource in NVIDIA container runtime
3. Initialize Docker Swarm on head node
4. Join workers to the swarm
5. Verify GPU resources are visible

### 5. Start the Cluster

From the **head node**, run:
```bash
./start_cluster.sh
```

This single command will:
1. Pull the Docker image on both nodes
2. Download the model (if not cached)
3. SSH to worker(s) and start TensorRT-LLM worker containers
4. Start TensorRT-LLM server on the head node
5. Wait for the cluster to become ready (~2-5 minutes)

### 6. Verify the Cluster

```bash
# Check health
curl http://localhost:8355/health

# List models
curl http://localhost:8355/v1/models

# Test inference
curl http://localhost:8355/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"nvidia/Qwen3-235B-A22B-FP4","messages":[{"role":"user","content":"Hello!"}],"max_tokens":50}'
```

### 7. Run Benchmarks

```bash
# Quick sanity test
./benchmark_current.sh quick

# Throughput test
./benchmark_current.sh throughput

# Custom benchmark
./benchmark_current.sh -n 100 -i 512 -o 256
```

### 8. Stop the Cluster

```bash
# Stop containers only (keeps Docker Swarm intact for faster restart)
./stop_cluster.sh

# Stop containers and tear down Docker Swarm
./stop_cluster.sh --teardown-swarm
```

## Scripts Overview

| Script | Description |
|--------|-------------|
| `setup-env.sh` | Interactive environment setup (source this!) |
| `config.env` | Configuration template |
| `setup_swarm.sh` | One-time Docker Swarm setup for multi-node GPU scheduling |
| `start_cluster.sh` | **Main script** - starts head + workers via SSH |
| `stop_cluster.sh` | Stops containers on head + workers (optionally tears down swarm) |
| `switch_model.sh` | Switch between different models |
| `benchmark_current.sh` | Benchmark current model |
| `benchmark_all.sh` | Benchmark all models and create comparison matrix |

## Configuration

Key settings in `config.env` or `config.local.env`:

```bash
# ┌─────────────────────────────────────────────────────────────────┐
# │ Required for Multi-Node                                         │
# └─────────────────────────────────────────────────────────────────┘
WORKER_HOST="192.168.7.111"        # Worker Ethernet IP for SSH access
WORKER_IB_IP="169.254.216.8"       # Worker InfiniBand IP for NCCL/RDMA
WORKER_USER="<username>"           # SSH username for workers
HEAD_IP="192.168.6.64"             # Head node Ethernet IP for Docker Swarm

# ┌─────────────────────────────────────────────────────────────────┐
# │ Model Settings                                                  │
# └─────────────────────────────────────────────────────────────────┘
MODEL="nvidia/Qwen3-235B-A22B-FP4" # Model to serve
TENSOR_PARALLEL="2"                # Total GPUs (1 per node × 2 nodes)
MAX_BATCH_SIZE="4"                 # Maximum batch size
MAX_NUM_TOKENS="32768"             # Maximum context length

# ┌─────────────────────────────────────────────────────────────────┐
# │ TensorRT-LLM Options                                            │
# └─────────────────────────────────────────────────────────────────┘
TRT_BACKEND="pytorch"              # Backend (pytorch recommended)
GPU_MEMORY_FRACTION="0.90"         # GPU memory fraction for KV cache
TRUST_REMOTE_CODE="true"           # For custom model code

# ┌─────────────────────────────────────────────────────────────────┐
# │ Optional                                                        │
# └─────────────────────────────────────────────────────────────────┘
HF_TOKEN="hf_xxx"                  # For gated models (Llama, etc.)
TRT_IMAGE="nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc4"  # Docker image
TRT_PORT="8355"                    # API port
```

### Finding Worker IP Addresses

On the **worker node**, find both IP addresses:

```bash
# ┌─────────────────────────────────────────────────────────────────┐
# │ 1. WORKER_HOST - Standard Ethernet IP (for SSH)                 │
# └─────────────────────────────────────────────────────────────────┘
# This is your regular network IP (e.g., 192.168.x.x or 10.x.x.x)
hostname -I | awk '{print $1}'
# Or check your network interface:
ip addr show eth0 | grep "inet "    # Replace eth0 with your interface

# ┌─────────────────────────────────────────────────────────────────┐
# │ 2. WORKER_IB_IP - InfiniBand IP (for NCCL/RDMA)                 │
# └─────────────────────────────────────────────────────────────────┘
# Find InfiniBand interface name
ibdev2netdev
# Example output: mlx5_0 port 1 ==> enp1s0f1np1 (Up)

# Get IP address for that interface (typically 169.254.x.x)
ip addr show enp1s0f1np1 | grep "inet "
# Example output: inet 169.254.216.8/16 ...
```

**Summary:**
- `WORKER_HOST` = Worker's standard Ethernet IP (e.g., `192.168.7.111`) - used for SSH
- `WORKER_IB_IP` = Worker's InfiniBand IP (e.g., `169.254.216.8`) - used for high-speed GPU communication
- `HEAD_IP` = Head node's standard Ethernet IP (e.g., `192.168.6.64`) - used for Docker Swarm

> **Note:** Docker Swarm should use standard Ethernet IPs (not InfiniBand link-local IPs) for reliability. NCCL will still use InfiniBand for GPU-to-GPU transfers regardless of the Swarm network.

## Switching Models

Use `switch_model.sh` to easily switch between models:

```bash
# List available models
./switch_model.sh --list

# Interactive selection
./switch_model.sh

# Direct selection (by number)
./switch_model.sh 3  # Switch to specific model

# Update config only (don't restart)
./switch_model.sh -s 5

# Download model only
./switch_model.sh -d 1

# Download and sync to worker
./switch_model.sh -r 1
```

## Supported Models

All models run across both DGX Spark nodes (TP=2) for maximum performance.

| # | Model | Size | Notes |
|---|-------|------|-------|
| 1 | `nvidia/Qwen3-235B-A22B-FP4` | ~60GB | Default, FP4 quantized, very fast |
| 2 | `openai/gpt-oss-120b` | ~80GB+ | MoE, reasoning model |
| 3 | `openai/gpt-oss-20b` | ~16-20GB | MoE, fast |
| 4 | `meta-llama/Llama-3.3-70B-Instruct` | ~65GB | High quality (needs HF token) |
| 5 | `Qwen/Qwen2.5-72B-Instruct` | ~70GB | High quality |
| 6 | `Qwen/Qwen2.5-32B-Instruct` | ~30GB | Strong mid-size |
| 7 | `mistralai/Mixtral-8x7B-Instruct-v0.1` | ~45GB | MoE, fast |

## Benchmark Profiles

The `benchmark_current.sh` script supports multiple profiles:

| Profile | Prompts | Input | Output | Use Case |
|---------|---------|-------|--------|----------|
| `quick` | 10 | 128 | 128 | Sanity test |
| `short` | 50 | 256 | 256 | Quick benchmark |
| `medium` | 100 | 512 | 512 | Standard benchmark |
| `long` | 200 | 1024 | 1024 | Extended test |
| `throughput` | 500 | 256 | 256 | Max throughput |
| `latency` | 100 | 128 | 128 | Rate-limited latency |
| `stress` | 1000 | 512 | 512 | Stress test |

```bash
# Run specific profile
./benchmark_current.sh throughput

# Custom settings
./benchmark_current.sh -n 200 -i 512 -o 1024 -c 32

# View results
cat benchmark_results/bench_*.json | python3 -m json.tool
```

### Benchmark All Models

Use `benchmark_all.sh` to automatically benchmark multiple models and create a comparison matrix:

```bash
# Benchmark all models (takes several hours)
./benchmark_all.sh

# Only single-node models (faster)
./benchmark_all.sh --single-node

# Skip models requiring HF token
./benchmark_all.sh --skip-token

# Quick benchmark of specific models
./benchmark_all.sh --models "1,2,3" --profile quick

# Dry run - see what would be benchmarked
./benchmark_all.sh --dry-run --single-node
```

The script generates:
- **Summary matrix** with throughput and latency for all models
- **CSV file** for spreadsheet analysis
- **JSON file** for programmatic access
- **Per-model benchmark files** with detailed metrics

## API Endpoints

Once running, the API is available on the head node:

| Endpoint | Description |
|----------|-------------|
| `http://<head-ip>:8355/health` | Health check |
| `http://<head-ip>:8355/v1/models` | List models |
| `http://<head-ip>:8355/v1/chat/completions` | Chat API (OpenAI compatible) |
| `http://<head-ip>:8355/v1/completions` | Completions API |

### Example: Chat Completion

```bash
curl http://localhost:8355/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Qwen3-235B-A22B-FP4",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain quantum computing briefly."}
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

### Example: Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8355/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="nvidia/Qwen3-235B-A22B-FP4",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)
print(response.choices[0].message.content)
```

## Troubleshooting

### Docker Swarm Setup Issues

```bash
# Check swarm status
docker info | grep Swarm

# Check node status
docker node ls

# Check GPU resources are visible
docker node inspect <node-name> --format '{{.Description.Resources.GenericResources}}'

# Re-run swarm setup if needed
./setup_swarm.sh
```

### SSH Connection Failed

```bash
# Test SSH connectivity
ssh <username>@<worker-ip> "hostname"

# If it fails, setup passwordless SSH:
ssh-copy-id <username>@<worker-ip>
```

### Worker Not Joining Cluster

```bash
# Check worker logs (from head node)
ssh <username>@<worker-ip> "docker logs trtllm-worker"

# Check Docker Swarm node status
docker node ls
```

### Low Throughput (Using Ethernet instead of InfiniBand)

```bash
# Check NCCL transport in logs
docker logs trtllm-head 2>&1 | grep -E "NCCL|NET"

# Good: "NCCL INFO NET/IB" or "GPU Direct RDMA"
# Bad:  "NCCL INFO NET/Socket" (falling back to Ethernet)

# Check InfiniBand devices
ibv_devinfo
ibdev2netdev  # Should show "(Up)" status
```

### NCCL Communication Issues

```bash
# Check InfiniBand devices
ibv_devinfo

# If IB issues persist, check cables and network config
ibstatus

# Try disabling IB as a test (not recommended for production)
export NCCL_IB_DISABLE=1
./start_cluster.sh
```

### Out of Memory

```bash
# Reduce memory fraction
export GPU_MEMORY_FRACTION=0.80
./start_cluster.sh

# Or reduce batch size
export MAX_BATCH_SIZE=2
./start_cluster.sh

# Or try a smaller model
./switch_model.sh --list  # Pick smaller model
```

### TensorRT-LLM Server Not Starting

```bash
# Check server logs
docker logs trtllm-head

# Common issues:
# - Insufficient GPUs for tensor-parallel-size
# - Model download failed (check HF_TOKEN for gated models)
# - NCCL timeout (check InfiniBand connectivity)
# - Docker Swarm not setup (run setup_swarm.sh)
```

### Model Download Issues

```bash
# Check if HF token is set (for gated models)
echo $HF_TOKEN

# Pre-download model manually
./switch_model.sh -d <model-number>

# Sync to worker
./switch_model.sh -r <model-number>
```

## Advanced Usage

### Start Head Only (Single Node)

```bash
./start_cluster.sh --head-only
```

### Skip Docker Pull (Faster Restart)

```bash
./start_cluster.sh --skip-pull
```

### Stop Local Only

```bash
./stop_cluster.sh --local-only
```

### Teardown Docker Swarm

```bash
# Stop cluster and remove swarm configuration
./stop_cluster.sh --teardown-swarm
```

### Force Stop Without Confirmation

```bash
./stop_cluster.sh -f
```

### View Container Logs in Real-Time

```bash
# Head node
docker logs -f trtllm-head

# Worker (from head via SSH)
ssh <worker-ip> "docker logs -f trtllm-worker"
```

## Performance Notes

### Expected Performance (Qwen3 235B FP4 on 2x DGX Spark)

| Metric | Value |
|--------|-------|
| Output Throughput | ~80-120 tok/s |
| Time to First Token | ~1-3s |
| Batch Throughput | ~500-800 tok/s |

### Optimization Tips

1. **Configure both IPs correctly:**
   - `WORKER_HOST` = Ethernet IP for SSH (e.g., 192.168.x.x)
   - `WORKER_IB_IP` = InfiniBand IP for NCCL (e.g., 169.254.x.x)
2. **Docker Swarm** - Must be setup for multi-node GPU scheduling
3. **Memory Fraction** - Set to 0.90 for max KV cache, reduce if OOM
4. **Pre-download Models** - Use `switch_model.sh -d` to avoid download delays
5. **PyTorch Backend** - Recommended for DGX Spark (`TRT_BACKEND=pytorch`)

## File Structure

```
trt-dgx-spark/
├── README.md              # This file
├── config.env             # Configuration template
├── config.local.env       # Your local config (gitignored)
├── docker-compose.yml     # Docker Compose configuration
├── setup-env.sh           # Interactive setup script
├── setup_swarm.sh         # Docker Swarm setup (one-time)
├── start_cluster.sh       # Main cluster startup script
├── stop_cluster.sh        # Cluster shutdown script
├── switch_model.sh        # Model switching utility
├── benchmark_current.sh   # Single model benchmark tool
├── benchmark_all.sh       # Multi-model comparison benchmark
└── benchmark_results/     # Benchmark output directory
```

## References

- [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/)
- [TensorRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM)
- [NVIDIA TensorRT-LLM Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt-llm)
- [NVIDIA DGX Spark Documentation](https://docs.nvidia.com/dgx-spark/)
- [NVIDIA NCCL over InfiniBand](https://build.nvidia.com/spark/nccl/stacked-sparks)
- [Docker Swarm GPU Support](https://docs.docker.com/engine/extend/plugins_volume/)

## License

MIT
