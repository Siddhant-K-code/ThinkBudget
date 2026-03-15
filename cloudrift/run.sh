#!/bin/bash
# ThinkBudget — CloudRift Quick Start
#
# Starts a vLLM inference server + ThinkBudget proxy on a CloudRift GPU.
#
# Usage:
#   # Default: DeepSeek-R1-Distill-Qwen-7B on RTX 4090
#   ./run.sh
#
#   # Custom model:
#   MODEL=Qwen/QwQ-32B ./run.sh
#
#   # Custom GPU cost (RTX 5090):
#   GPU_COST=0.65 GPU_NAME="RTX 5090" ./run.sh

set -euo pipefail

MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
GPU_COST="${GPU_COST:-0.39}"
GPU_NAME="${GPU_NAME:-RTX 4090}"
VLLM_PORT="${VLLM_PORT:-8000}"
PROXY_PORT="${PROXY_PORT:-9100}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║                ThinkBudget on CloudRift                  ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Model:     ${MODEL}"
echo "║  GPU:       ${GPU_NAME} (\$${GPU_COST}/hr)"
echo "║  vLLM:      port ${VLLM_PORT}"
echo "║  Proxy:     port ${PROXY_PORT}"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Check GPU
echo "[1/4] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
    echo "ERROR: No GPU detected. Are you running on a CloudRift GPU instance?"
    exit 1
}
echo ""

# Start vLLM in background
echo "[2/4] Starting vLLM inference server..."
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$VLLM_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    &
VLLM_PID=$!

# Wait for vLLM to be ready
echo "     Waiting for vLLM to load model..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "     vLLM ready!"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM process died. Check logs above."
        exit 1
    fi
    sleep 2
done
echo ""

# Generate benchmark datasets if not present
echo "[3/4] Preparing benchmark datasets..."
if [ ! -f benchmarks/datasets/trivial.jsonl ]; then
    cd benchmarks/datasets && python3 generate_datasets.py && cd ../..
fi
echo ""

# Start ThinkBudget proxy
echo "[4/4] Starting ThinkBudget proxy..."
python3 -m thinkbudget.cli serve \
    --backend-url "http://localhost:${VLLM_PORT}" \
    --model "$MODEL" \
    --port "$PROXY_PORT" \
    --gpu-cost "$GPU_COST" \
    --gpu-name "$GPU_NAME" \
    &
PROXY_PID=$!

# Wait for proxy
sleep 3
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ThinkBudget is running!"
echo ""
echo "  Proxy:      http://localhost:${PROXY_PORT}"
echo "  Dashboard:  http://localhost:${PROXY_PORT}/dashboard"
echo "  vLLM:       http://localhost:${VLLM_PORT}"
echo ""
echo "  Quick test:"
echo "    curl http://localhost:${PROXY_PORT}/v1/chat/completions \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"model\": \"${MODEL}\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'"
echo ""
echo "  Run benchmarks:"
echo "    python3 benchmarks/run_benchmark.py --mode compare \\"
echo "      --backend-url http://localhost:${VLLM_PORT} \\"
echo "      --proxy-url http://localhost:${PROXY_PORT} \\"
echo "      --model ${MODEL}"
echo "════════════════════════════════════════════════════════════"

# Keep running
wait $VLLM_PID $PROXY_PID
