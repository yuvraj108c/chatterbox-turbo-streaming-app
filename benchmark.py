#!/usr/bin/env python3
"""
Benchmark script for Chatterbox TTS server.
Tests RTF (Real-Time Factor) with various concurrency levels to find optimal throughput.

Usage:
    # First, start the server:
    python server.py
    
    # Register a voice:
    python benchmark.py --register-voice path/to/voice.wav
    
    # Run benchmark:
    python benchmark.py --voice-id <voice_id> --concurrency 1,2,4,8,16 --requests 50
    
    # Find max concurrency with RTF < 1:
    python benchmark.py --voice-id <voice_id> --find-max-concurrency
"""

import asyncio
import argparse
import time
import statistics
import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path

import httpx

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_SERVER_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 120.0

# Test texts of varying lengths
TEST_TEXTS = [
    # Short (~2s audio)
    "Hello, how are you doing today?",
    
    # Medium (~5s audio)
    "Oh, that's hilarious! [chuckle] Um anyway, how are you doing today? I hope everything is going well with you.",
    
    # Long (~10s audio)
    "Welcome to this demonstration of our text-to-speech system. This is a longer piece of text that will help us understand how the system performs with more substantial content. The goal is to measure both quality and performance.",
    
    # Very long (~20s audio)
    "In the realm of artificial intelligence, text-to-speech synthesis has made remarkable progress over the past few years. Modern neural network based approaches can produce speech that is nearly indistinguishable from human recordings. This technology has applications ranging from accessibility tools for visually impaired users, to virtual assistants, audiobook narration, and content creation. The challenge now lies in making these systems fast enough for real-time applications while maintaining high quality.",
]


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class RequestResult:
    """Result of a single TTS request."""
    success: bool
    text_length: int
    audio_duration: float
    generation_time: float
    rtf: float
    tokens_generated: int
    error: Optional[str] = None


@dataclass
class ConcurrencyResult:
    """Results for a specific concurrency level."""
    concurrency: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    
    # RTF stats
    rtf_mean: float
    rtf_median: float
    rtf_p95: float
    rtf_min: float
    rtf_max: float
    
    # Timing stats
    total_time: float
    throughput_rps: float  # Requests per second
    throughput_audio_per_sec: float  # Audio seconds generated per wall-clock second
    
    # Latency stats
    latency_mean: float
    latency_p95: float


@dataclass
class BenchmarkReport:
    """Full benchmark report."""
    server_url: str
    voice_id: str
    test_texts_count: int
    timestamp: str
    results: List[ConcurrencyResult]
    optimal_concurrency: int
    max_concurrency_rtf_under_1: int


# ============================================================================
# Benchmark Client
# ============================================================================

class BenchmarkClient:
    """Async client for benchmarking the TTS server."""
    
    def __init__(self, server_url: str = DEFAULT_SERVER_URL):
        self.server_url = server_url.rstrip("/")
        self.client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)
        return self
    
    async def __aexit__(self, *args):
        if self.client:
            await self.client.aclose()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check server health."""
        response = await self.client.get(f"{self.server_url}/health")
        response.raise_for_status()
        return response.json()
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get server metrics."""
        response = await self.client.get(f"{self.server_url}/metrics")
        response.raise_for_status()
        return response.json()
    
    async def register_voice(self, voice_path: str, voice_id: Optional[str] = None) -> str:
        """Register a voice file."""
        with open(voice_path, "rb") as f:
            files = {"voice_file": (Path(voice_path).name, f, "audio/wav")}
            data = {}
            if voice_id:
                data["voice_id"] = voice_id
            
            response = await self.client.post(
                f"{self.server_url}/voices/register",
                files=files,
                data=data,
            )
            response.raise_for_status()
            return response.json()["voice_id"]
    
    async def list_voices(self) -> List[str]:
        """List registered voices."""
        response = await self.client.get(f"{self.server_url}/voices")
        response.raise_for_status()
        return response.json()
    
    async def synthesize(self, text: str, voice_id: str) -> RequestResult:
        """
        Synthesize text and return metrics.
        Uses the JSON endpoint for faster benchmarking (no audio transfer).
        """
        start_time = time.perf_counter()
        
        try:
            response = await self.client.post(
                f"{self.server_url}/tts/json",
                json={"text": text, "voice_id": voice_id},
            )
            response.raise_for_status()
            data = response.json()
            
            return RequestResult(
                success=True,
                text_length=len(text),
                audio_duration=data["audio_duration"],
                generation_time=data["generation_time"],
                rtf=data["rtf"],
                tokens_generated=data["tokens_generated"],
            )
        
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            return RequestResult(
                success=False,
                text_length=len(text),
                audio_duration=0.0,
                generation_time=elapsed,
                rtf=float("inf"),
                tokens_generated=0,
                error=str(e),
            )
    
    async def synthesize_with_audio(
        self, text: str, voice_id: str
    ) -> tuple[RequestResult, Optional[bytes]]:
        """Synthesize and return audio bytes (for validation)."""
        start_time = time.perf_counter()
        
        try:
            response = await self.client.post(
                f"{self.server_url}/tts",
                data={"text": text, "voice_id": voice_id, "stream": "false"},
            )
            response.raise_for_status()
            
            audio_bytes = response.content
            
            # Parse headers for metrics
            audio_duration = float(response.headers.get("X-Audio-Duration", 0))
            generation_time = float(response.headers.get("X-Generation-Time", 0))
            rtf = float(response.headers.get("X-RTF", float("inf")))
            tokens_generated = int(response.headers.get("X-Tokens-Generated", 0))
            
            return RequestResult(
                success=True,
                text_length=len(text),
                audio_duration=audio_duration,
                generation_time=generation_time,
                rtf=rtf,
                tokens_generated=tokens_generated,
            ), audio_bytes
        
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            return RequestResult(
                success=False,
                text_length=len(text),
                audio_duration=0.0,
                generation_time=elapsed,
                rtf=float("inf"),
                tokens_generated=0,
                error=str(e),
            ), None


# ============================================================================
# Benchmark Functions
# ============================================================================

async def run_concurrent_requests(
    client: BenchmarkClient,
    voice_id: str,
    texts: List[str],
    num_requests: int,
    concurrency: int,
) -> ConcurrencyResult:
    """Run benchmark with specified concurrency level."""
    print(f"\n{'='*60}")
    print(f"Testing concurrency={concurrency} with {num_requests} requests")
    print(f"{'='*60}")
    
    semaphore = asyncio.Semaphore(concurrency)
    results: List[RequestResult] = []
    
    async def bounded_request(text: str) -> RequestResult:
        async with semaphore:
            return await client.synthesize(text, voice_id)
    
    # Create requests with varied text lengths
    tasks = []
    for i in range(num_requests):
        text = texts[i % len(texts)]
        tasks.append(bounded_request(text))
    
    # Run all requests
    start_time = time.perf_counter()
    results = await asyncio.gather(*tasks)
    total_time = time.perf_counter() - start_time
    
    # Compute statistics
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    if not successful:
        print(f"  All {len(failed)} requests failed!")
        return ConcurrencyResult(
            concurrency=concurrency,
            total_requests=num_requests,
            successful_requests=0,
            failed_requests=len(failed),
            rtf_mean=float("inf"),
            rtf_median=float("inf"),
            rtf_p95=float("inf"),
            rtf_min=float("inf"),
            rtf_max=float("inf"),
            total_time=total_time,
            throughput_rps=0,
            throughput_audio_per_sec=0,
            latency_mean=float("inf"),
            latency_p95=float("inf"),
        )
    
    rtfs = [r.rtf for r in successful]
    latencies = [r.generation_time for r in successful]
    audio_durations = [r.audio_duration for r in successful]
    
    rtf_sorted = sorted(rtfs)
    latency_sorted = sorted(latencies)
    
    result = ConcurrencyResult(
        concurrency=concurrency,
        total_requests=num_requests,
        successful_requests=len(successful),
        failed_requests=len(failed),
        rtf_mean=statistics.mean(rtfs),
        rtf_median=statistics.median(rtfs),
        rtf_p95=rtf_sorted[int(len(rtf_sorted) * 0.95)] if len(rtf_sorted) > 1 else rtfs[0],
        rtf_min=min(rtfs),
        rtf_max=max(rtfs),
        total_time=total_time,
        throughput_rps=len(successful) / total_time,
        throughput_audio_per_sec=sum(audio_durations) / total_time,
        latency_mean=statistics.mean(latencies),
        latency_p95=latency_sorted[int(len(latency_sorted) * 0.95)] if len(latency_sorted) > 1 else latencies[0],
    )
    
    # Print results
    print(f"  Successful: {result.successful_requests}/{result.total_requests}")
    print(f"  RTF - Mean: {result.rtf_mean:.3f}, Median: {result.rtf_median:.3f}, P95: {result.rtf_p95:.3f}")
    print(f"  RTF - Min: {result.rtf_min:.3f}, Max: {result.rtf_max:.3f}")
    print(f"  Latency - Mean: {result.latency_mean:.2f}s, P95: {result.latency_p95:.2f}s")
    print(f"  Throughput: {result.throughput_rps:.2f} req/s, {result.throughput_audio_per_sec:.2f}x realtime")
    print(f"  Total time: {result.total_time:.2f}s")
    
    return result


async def find_max_concurrency(
    client: BenchmarkClient,
    voice_id: str,
    texts: List[str],
    requests_per_level: int = 20,
    rtf_threshold: float = 1.0,
    start_concurrency: int = 1,
    max_concurrency: int = 32,
) -> int:
    """Binary search to find max concurrency with RTF < threshold."""
    print(f"\n{'#'*60}")
    print(f"Finding max concurrency with P95 RTF < {rtf_threshold}")
    print(f"{'#'*60}")
    
    low, high = start_concurrency, max_concurrency
    best_concurrency = start_concurrency
    
    while low <= high:
        mid = (low + high) // 2
        result = await run_concurrent_requests(
            client, voice_id, texts, requests_per_level, mid
        )
        
        if result.rtf_p95 < rtf_threshold:
            best_concurrency = mid
            low = mid + 1
            print(f"  -> RTF P95 {result.rtf_p95:.3f} < {rtf_threshold}, trying higher concurrency")
        else:
            high = mid - 1
            print(f"  -> RTF P95 {result.rtf_p95:.3f} >= {rtf_threshold}, trying lower concurrency")
    
    print(f"\n{'='*60}")
    print(f"Maximum concurrency with RTF < {rtf_threshold}: {best_concurrency}")
    print(f"{'='*60}")
    
    return best_concurrency


async def run_benchmark(
    server_url: str,
    voice_id: str,
    concurrency_levels: List[int],
    requests_per_level: int,
    texts: List[str],
) -> BenchmarkReport:
    """Run full benchmark suite."""
    async with BenchmarkClient(server_url) as client:
        # Health check
        print("Checking server health...")
        health = await client.health_check()
        print(f"  Status: {health['status']}")
        print(f"  GPU Provider: {health['gpu_provider']}")
        print(f"  Model dtype: {health['model_dtype']}")
        
        # Verify voice
        voices = await client.list_voices()
        if voice_id not in voices:
            raise ValueError(f"Voice ID '{voice_id}' not found. Available: {voices}")
        
        # Warmup
        print("\nWarming up (3 requests)...")
        for _ in range(3):
            await client.synthesize(texts[0], voice_id)
        
        # Run benchmarks
        results = []
        for concurrency in concurrency_levels:
            result = await run_concurrent_requests(
                client, voice_id, texts, requests_per_level, concurrency
            )
            results.append(result)
        
        # Find optimal
        optimal = min(results, key=lambda r: r.rtf_mean)
        max_under_1 = max(
            (r for r in results if r.rtf_p95 < 1.0),
            key=lambda r: r.concurrency,
            default=results[0],
        )
        
        report = BenchmarkReport(
            server_url=server_url,
            voice_id=voice_id,
            test_texts_count=len(texts),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            results=results,
            optimal_concurrency=optimal.concurrency,
            max_concurrency_rtf_under_1=max_under_1.concurrency,
        )
        
        return report


def print_report(report: BenchmarkReport):
    """Print benchmark report summary."""
    print(f"\n{'#'*60}")
    print("BENCHMARK REPORT")
    print(f"{'#'*60}")
    print(f"Server: {report.server_url}")
    print(f"Voice ID: {report.voice_id}")
    print(f"Timestamp: {report.timestamp}")
    print(f"\nOptimal concurrency (lowest mean RTF): {report.optimal_concurrency}")
    print(f"Max concurrency with RTF < 1.0: {report.max_concurrency_rtf_under_1}")
    
    print(f"\n{'='*80}")
    print(f"{'Conc':>6} | {'RTF Mean':>10} | {'RTF P95':>10} | {'Throughput':>12} | {'Audio/s':>10}")
    print(f"{'='*80}")
    
    for r in report.results:
        print(
            f"{r.concurrency:>6} | "
            f"{r.rtf_mean:>10.3f} | "
            f"{r.rtf_p95:>10.3f} | "
            f"{r.throughput_rps:>10.2f}/s | "
            f"{r.throughput_audio_per_sec:>8.2f}x"
        )
    
    print(f"{'='*80}")


# ============================================================================
# CLI
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Benchmark Chatterbox TTS Server")
    parser.add_argument(
        "--server-url",
        default=DEFAULT_SERVER_URL,
        help="TTS server URL",
    )
    parser.add_argument(
        "--register-voice",
        help="Path to voice file to register",
    )
    parser.add_argument(
        "--voice-id",
        help="Voice ID to use for benchmarking",
    )
    parser.add_argument(
        "--concurrency",
        default="1,2,4,8",
        help="Comma-separated concurrency levels to test",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=20,
        help="Number of requests per concurrency level",
    )
    parser.add_argument(
        "--find-max-concurrency",
        action="store_true",
        help="Find max concurrency with RTF < 1",
    )
    parser.add_argument(
        "--output",
        help="Output JSON file for results",
    )
    
    args = parser.parse_args()
    
    async with BenchmarkClient(args.server_url) as client:
        # Register voice if requested
        if args.register_voice:
            print(f"Registering voice from {args.register_voice}...")
            voice_id = await client.register_voice(args.register_voice)
            print(f"Voice registered with ID: {voice_id}")
            
            if not args.voice_id:
                args.voice_id = voice_id
        
        # List voices
        voices = await client.list_voices()
        print(f"Available voices: {voices}")
        
        if not args.voice_id:
            if voices:
                args.voice_id = voices[0]
                print(f"Using first available voice: {args.voice_id}")
            else:
                print("No voices available. Register a voice first with --register-voice")
                return
        
        # Find max concurrency
        if args.find_max_concurrency:
            max_conc = await find_max_concurrency(
                client,
                args.voice_id,
                TEST_TEXTS,
                requests_per_level=args.requests,
            )
            print(f"\nResult: Max concurrency with RTF < 1.0 is {max_conc}")
            return
        
        # Run benchmark
        concurrency_levels = [int(x) for x in args.concurrency.split(",")]
        
        report = await run_benchmark(
            args.server_url,
            args.voice_id,
            concurrency_levels,
            args.requests,
            TEST_TEXTS,
        )
        
        print_report(report)
        
        # Save results
        if args.output:
            with open(args.output, "w") as f:
                json.dump(asdict(report), f, indent=2)
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
