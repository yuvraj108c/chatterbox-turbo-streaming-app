# Chatterbox TTS Server

GPU-accelerated Text-to-Speech API using Chatterbox Turbo ONNX models, optimized for L40S GPU.

## Features

- **FastAPI server** with streaming audio response
- **GPU inference** via ONNX Runtime CUDA provider
- **FP16 support** for 2x throughput on L40S tensor cores
- **Voice caching** - pre-encode reference voices for faster synthesis
- **Concurrency control** - configurable limits with backpressure
- **RTF benchmarking** - measure Real-Time Factor under load

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the server

```bash
./run.sh
# Or manually:
python server.py
```

### 3. Register a voice

```bash
curl -X POST http://localhost:8000/voices/register \
  -F "voice_file=@/path/to/voice.wav"
# Returns: {"voice_id": "abc123...", "message": "Voice registered successfully"}
```

### 4. Synthesize speech

```bash
# Streaming audio response
curl -X POST http://localhost:8000/tts \
  -F "text=Hello, how are you today?" \
  -F "voice_id=abc123" \
  --output output.wav

# With uploaded voice file (no pre-registration)
curl -X POST http://localhost:8000/tts \
  -F "text=Hello, how are you today?" \
  -F "voice_file=@/path/to/voice.wav" \
  --output output.wav
```

## Performance

### GPU Optimization

The server uses optimized **CUDA** settings for L40S GPU:
- **FP16 precision**: Optimized for tensor cores
- **Optimized cuDNN**: HEURISTIC algo search, increased workspace for Conv operations
- **Dynamic memory**: Prevents BFC arena fragmentation

> **Note**: Conv operations may show fallback warnings but performance is still excellent (RTF ~0.19). For even better performance, install TensorRT (eliminates all fallbacks).

### Typical Performance (L40S GPU)

- **RTF**: ~0.19 (5.2x realtime)  
- **Throughput**: 0.4-0.5 req/s per worker
- **Latency**: ~2-3s for 10-15s audio
- **Concurrency**: 2-4 concurrent requests optimal

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/metrics` | GET | Server metrics (RTF, throughput, etc.) |
| `/voices` | GET | List registered voice IDs |
| `/voices/register` | POST | Register a reference voice |
| `/tts` | POST | Synthesize speech (returns audio) |
| `/tts/json` | POST | Synthesize speech (returns metadata only) |

## Benchmarking

### Run benchmark suite

```bash
# Register voice and run benchmarks
./run_benchmark.sh /path/to/voice.wav

# Or manually:
python benchmark.py --register-voice /path/to/voice.wav
python benchmark.py --voice-id <voice_id> --concurrency 1,2,4,8,16 --requests 50
```

### Find max concurrency with RTF < 1

```bash
python benchmark.py --voice-id <voice_id> --find-max-concurrency
```

### Example output

```
==============================================================
BENCHMARK REPORT
==============================================================
Server: http://localhost:8000
Voice ID: abc123
Timestamp: 2025-12-26 10:30:00

Optimal concurrency (lowest mean RTF): 4
Max concurrency with RTF < 1.0: 8

================================================================================
  Conc |   RTF Mean |    RTF P95 |   Throughput |    Audio/s
================================================================================
     1 |      0.312 |      0.358 |       3.21/s |      9.6x
     2 |      0.341 |      0.402 |       5.87/s |     17.5x
     4 |      0.398 |      0.487 |      10.05/s |     30.2x
     8 |      0.521 |      0.673 |      15.36/s |     46.1x
    16 |      0.842 |      1.124 |      18.99/s |     56.9x
================================================================================
```

## Configuration

### Server Config ([server.py](server.py))

```python
class ServerConfig:
    MAX_CONCURRENT_REQUESTS: int = 8   # Adjust based on RTF testing
    THREAD_POOL_SIZE: int = 8
    MAX_TEXT_LENGTH: int = 5000
    MAX_VOICE_FILE_SIZE: int = 50 MB
    REQUEST_TIMEOUT: float = 120.0
    MODEL_DTYPE: str = "fp16"          # fp32, fp16, q8, q4, q4f16
```

### TTS Engine Config ([tts_engine.py](tts_engine.py))

```python
class TTSConfig:
    model_dtype: str = "fp16"
    max_new_tokens: int = 1024
    repetition_penalty: float = 1.2
    gpu_device_id: int = 0
    gpu_mem_limit_gb: int = 40         # L40S has 48GB
    voice_cache_size: int = 100
    apply_watermark: bool = False
```

## L40S GPU Optimization

The server is optimized for NVIDIA L40S (48GB VRAM):

1. **FP16 inference** - Uses tensor cores for ~2x throughput
2. **Memory management** - 40GB limit leaves headroom for CUDA context
3. **Session options** - Graph optimization, memory reuse enabled
4. **Single worker** - Avoids GPU memory fragmentation

### Expected Performance (L40S)

| Concurrency | RTF (P95) | Throughput | Notes |
|-------------|-----------|------------|-------|
| 1 | ~0.35 | ~3 req/s | Low latency |
| 4 | ~0.50 | ~10 req/s | Good balance |
| 8 | ~0.70 | ~15 req/s | High throughput |
| 12+ | ~1.0+ | ~18 req/s | RTF exceeds 1 |

**Recommendation**: Set `MAX_CONCURRENT_REQUESTS=8` for optimal RTF < 1 operation.

## File Structure

```
chatterbox/
├── main.py              # Original standalone script
├── tts_engine.py        # TTSEngine class with GPU optimization
├── server.py            # FastAPI server
├── benchmark.py         # RTF & concurrency benchmarking
├── requirements.txt     # Python dependencies
├── run.sh               # Server startup script
├── run_benchmark.sh     # Benchmark runner script
└── README.md            # This file
```

## Troubleshooting

### CUDA not available

```bash
# Check ONNX Runtime providers
python -c "import onnxruntime as ort; print(ort.get_available_providers())"

# Should show: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

### Out of memory

1. Reduce `MAX_CONCURRENT_REQUESTS` in [server.py](server.py)
2. Use quantized models: `MODEL_DTYPE = "q8"` or `"q4"`
3. Reduce `gpu_mem_limit_gb` in config

### High RTF under load

1. Run benchmark to find optimal concurrency: `python benchmark.py --find-max-concurrency`
2. Set `MAX_CONCURRENT_REQUESTS` to the discovered value
3. Consider using FP16 models if using FP32
