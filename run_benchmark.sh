#!/bin/bash
#
# Benchmark runner script for Chatterbox TTS
#

set -e

SERVER_URL="${SERVER_URL:-http://localhost:8000}"
VOICE_FILE="${1:-}"
OUTPUT_FILE="${OUTPUT_FILE:-benchmark_results.json}"

echo "=============================================="
echo "Chatterbox TTS Benchmark"
echo "=============================================="
echo "Server: $SERVER_URL"
echo ""

# Check server health
echo "Checking server health..."
curl -s "$SERVER_URL/health" | python3 -m json.tool
echo ""

# Register voice if provided
if [ -n "$VOICE_FILE" ]; then
    echo "Registering voice from: $VOICE_FILE"
    VOICE_ID=$(curl -s -X POST "$SERVER_URL/voices/register" \
        -F "voice_file=@$VOICE_FILE" | python3 -c "import sys, json; print(json.load(sys.stdin)['voice_id'])")
    echo "Voice ID: $VOICE_ID"
else
    # Get first available voice
    VOICE_ID=$(curl -s "$SERVER_URL/voices" | python3 -c "import sys, json; voices=json.load(sys.stdin); print(voices[0] if voices else '')")
    if [ -z "$VOICE_ID" ]; then
        echo "Error: No voices registered. Provide a voice file as argument."
        echo "Usage: $0 /path/to/voice.wav"
        exit 1
    fi
    echo "Using existing voice: $VOICE_ID"
fi

echo ""
echo "=============================================="
echo "Running benchmarks..."
echo "=============================================="

# Run comprehensive benchmark
python3 benchmark.py \
    --server-url "$SERVER_URL" \
    --voice-id "$VOICE_ID" \
    --concurrency "1,2,4,6,8,10,12,16" \
    --requests 30 \
    --output "$OUTPUT_FILE"

echo ""
echo "=============================================="
echo "Finding max concurrency with RTF < 1.0"
echo "=============================================="

python3 benchmark.py \
    --server-url "$SERVER_URL" \
    --voice-id "$VOICE_ID" \
    --find-max-concurrency \
    --requests 25

echo ""
echo "=============================================="
echo "Benchmark complete!"
echo "Results saved to: $OUTPUT_FILE"
echo "=============================================="

# Print final metrics
echo ""
echo "Server metrics:"
curl -s "$SERVER_URL/metrics" | python3 -m json.tool
