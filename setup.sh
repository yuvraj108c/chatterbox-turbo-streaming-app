#!/bin/bash
#
# Chatterbox TTS Server Startup Script
# Optimized for L40S GPU
#

set -e

# Configuration
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"  # Keep at 1 for GPU memory sharing
LOG_LEVEL="${LOG_LEVEL:-info}"

# CUDA Configuration for L40S
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_MODULE_LOADING=LAZY

# ONNX Runtime optimizations
export ORT_DISABLE_STATIC_ANALYSIS=1
export ORT_TENSORRT_FP16_ENABLE=1

# Memory optimizations
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

echo "=============================================="
echo "Chatterbox TTS Server"
echo "=============================================="
echo "Host: $HOST"
echo "Port: $PORT"
echo "Workers: $WORKERS"
echo "CUDA Device: $CUDA_VISIBLE_DEVICES"
echo "=============================================="

# Check CUDA availability
python3 -c "
import onnxruntime as ort
providers = ort.get_available_providers()
print(f'Available ONNX Runtime providers: {providers}')
if 'CUDAExecutionProvider' in providers:
    print('✓ CUDA is available')
else:
    print('⚠ CUDA not available, falling back to CPU')
"

# # Start server
# echo ""
# echo "Starting server..."
# exec uvicorn server:app \
#     --host "$HOST" \
#     --port "$PORT" \
#     --workers "$WORKERS" \
#     --log-level "$LOG_LEVEL" \
#     --timeout-keep-alive 120
