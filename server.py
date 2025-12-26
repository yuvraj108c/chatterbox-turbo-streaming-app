"""
FastAPI server for Chatterbox TTS with GPU streaming support.
Optimized for L40S GPU with concurrency control and RTF monitoring.
"""

import asyncio
import time
import uuid
import hashlib
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

from tts_engine import TTSEngine, TTSConfig, TTSResult, get_engine, SAMPLE_RATE

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class ServerConfig:
    """Server configuration."""
    # Concurrency settings
    # GPU inference must be serialized - ONNX BFC arena doesn't handle concurrent access well
    MAX_CONCURRENT_REQUESTS: int = 16  # Max requests allowed in queue
    GPU_THREAD_POOL_SIZE: int = 1  # MUST be 1 for GPU inference serialization
    
    # Request limits
    MAX_TEXT_LENGTH: int = 5000
    MAX_VOICE_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    
    # Timeouts
    REQUEST_TIMEOUT: float = 120.0  # seconds
    
    # Model settings
    MODEL_DTYPE: str = "fp16"  # fp32, fp16, q8, q4, q4f16


# ============================================================================
# Global State
# ============================================================================

# Concurrency control
_semaphore: Optional[asyncio.Semaphore] = None
_gpu_executor: Optional[ThreadPoolExecutor] = None  # Single thread for GPU ops
_engine: Optional[TTSEngine] = None

# Metrics
_metrics: Dict[str, Any] = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "total_audio_seconds": 0.0,
    "total_generation_time": 0.0,
    "rtf_samples": [],
    "concurrent_requests": 0,
    "peak_concurrent": 0,
}
_metrics_lock = asyncio.Lock()


# ============================================================================
# Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    global _semaphore, _gpu_executor, _engine
    
    logger.info("Starting TTS server...")
    
    # Initialize concurrency controls
    _semaphore = asyncio.Semaphore(ServerConfig.MAX_CONCURRENT_REQUESTS)
    # CRITICAL: Single thread executor to serialize GPU operations
    _gpu_executor = ThreadPoolExecutor(max_workers=ServerConfig.GPU_THREAD_POOL_SIZE)
    
    # Initialize TTS engine
    config = TTSConfig(
        model_dtype=ServerConfig.MODEL_DTYPE,
        gpu_mem_limit_gb=40,  # Not used with dynamic allocation
    )
    _engine = get_engine(config)
    
    logger.info(f"TTS server ready with max_concurrent={ServerConfig.MAX_CONCURRENT_REQUESTS}, gpu_threads={ServerConfig.GPU_THREAD_POOL_SIZE}")
    
    yield
    
    # Cleanup
    logger.info("Shutting down TTS server...")
    _gpu_executor.shutdown(wait=True)


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Chatterbox TTS API",
    description="GPU-accelerated Text-to-Speech API with streaming support",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request/Response Models
# ============================================================================

class TTSRequest(BaseModel):
    """TTS synthesis request."""
    text: str = Field(..., max_length=5000, description="Text to synthesize")
    voice_id: Optional[str] = Field(None, description="Pre-registered voice ID")
    stream: bool = Field(True, description="Stream audio response")


class TTSResponse(BaseModel):
    """TTS synthesis response metadata."""
    request_id: str
    audio_duration: float
    generation_time: float
    rtf: float
    tokens_generated: int
    sample_rate: int


class VoiceRegisterResponse(BaseModel):
    """Voice registration response."""
    voice_id: str
    message: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    gpu_provider: str
    model_dtype: str
    voice_cache_size: int


class MetricsResponse(BaseModel):
    """Server metrics response."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_audio_seconds: float
    total_generation_time: float
    average_rtf: float
    concurrent_requests: int
    peak_concurrent: int
    max_concurrent_allowed: int


# ============================================================================
# Helper Functions
# ============================================================================

async def update_metrics(result: Optional[TTSResult] = None, success: bool = True):
    """Update server metrics."""
    async with _metrics_lock:
        _metrics["total_requests"] += 1
        if success and result:
            _metrics["successful_requests"] += 1
            _metrics["total_audio_seconds"] += result.audio_duration
            _metrics["total_generation_time"] += result.generation_time
            _metrics["rtf_samples"].append(result.rtf)
            # Keep last 1000 samples
            if len(_metrics["rtf_samples"]) > 1000:
                _metrics["rtf_samples"] = _metrics["rtf_samples"][-1000:]
        else:
            _metrics["failed_requests"] += 1


async def track_concurrent(increment: bool):
    """Track concurrent request count."""
    async with _metrics_lock:
        if increment:
            _metrics["concurrent_requests"] += 1
            _metrics["peak_concurrent"] = max(
                _metrics["peak_concurrent"],
                _metrics["concurrent_requests"]
            )
        else:
            _metrics["concurrent_requests"] -= 1


def compute_voice_id(audio_bytes: bytes) -> str:
    """Compute a unique voice ID from audio bytes."""
    return hashlib.sha256(audio_bytes).hexdigest()[:16]


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    stats = _engine.get_stats()
    return HealthResponse(
        status="healthy",
        gpu_provider=stats["providers"][0] if stats["providers"] else "unknown",
        model_dtype=stats["model_dtype"],
        voice_cache_size=stats["voice_cache_size"],
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get server metrics."""
    async with _metrics_lock:
        avg_rtf = (
            sum(_metrics["rtf_samples"]) / len(_metrics["rtf_samples"])
            if _metrics["rtf_samples"]
            else 0.0
        )
        return MetricsResponse(
            total_requests=_metrics["total_requests"],
            successful_requests=_metrics["successful_requests"],
            failed_requests=_metrics["failed_requests"],
            total_audio_seconds=_metrics["total_audio_seconds"],
            total_generation_time=_metrics["total_generation_time"],
            average_rtf=avg_rtf,
            concurrent_requests=_metrics["concurrent_requests"],
            peak_concurrent=_metrics["peak_concurrent"],
            max_concurrent_allowed=ServerConfig.MAX_CONCURRENT_REQUESTS,
        )


@app.post("/voices/register", response_model=VoiceRegisterResponse)
async def register_voice(
    voice_file: UploadFile = File(..., description="Reference voice audio file"),
    voice_id: Optional[str] = Form(None, description="Custom voice ID (optional)"),
):
    """
    Register a reference voice for TTS.
    Pre-encodes the voice embedding for faster synthesis.
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    # Read voice file
    audio_bytes = await voice_file.read()
    
    if len(audio_bytes) > ServerConfig.MAX_VOICE_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Voice file too large. Max size: {ServerConfig.MAX_VOICE_FILE_SIZE} bytes"
        )
    
    # Compute or use provided voice ID
    vid = voice_id or compute_voice_id(audio_bytes)
    
    # Encode voice (runs in GPU executor - single threaded)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        _gpu_executor,
        _engine.encode_voice_from_bytes,
        audio_bytes,
        vid,
    )
    
    return VoiceRegisterResponse(
        voice_id=vid,
        message="Voice registered successfully",
    )


@app.get("/voices", response_model=List[str])
async def list_voices():
    """List registered voice IDs."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    return list(_engine._voice_cache.keys())


@app.post("/tts")
async def synthesize_speech(
    text: str = Form(..., description="Text to synthesize"),
    voice_id: Optional[str] = Form(None, description="Pre-registered voice ID"),
    voice_file: Optional[UploadFile] = File(None, description="Reference voice file (if no voice_id)"),
    stream: bool = Form(True, description="Stream audio response"),
):
    """
    Synthesize speech from text.
    
    Either provide a pre-registered voice_id or upload a voice_file.
    Response is streamed as audio/wav.
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    if not voice_id and not voice_file:
        raise HTTPException(
            status_code=400,
            detail="Either voice_id or voice_file must be provided"
        )
    
    if len(text) > ServerConfig.MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=413,
            detail=f"Text too long. Max length: {ServerConfig.MAX_TEXT_LENGTH}"
        )
    
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] TTS request: {len(text)} chars, voice_id={voice_id}")
    
    # Acquire semaphore for concurrency control
    try:
        async with asyncio.timeout(ServerConfig.REQUEST_TIMEOUT):
            await _semaphore.acquire()
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="Server too busy, try again later"
        )
    
    await track_concurrent(True)
    
    try:
        # Prepare voice
        voice_bytes = None
        cache_key = voice_id
        
        if voice_file:
            voice_bytes = await voice_file.read()
            if not voice_id:
                cache_key = compute_voice_id(voice_bytes)
        
        # Run synthesis in single-thread GPU executor
        loop = asyncio.get_event_loop()
        
        def _synthesize():
            if voice_bytes:
                return _engine.synthesize(
                    text=text,
                    voice_bytes=voice_bytes,
                    voice_cache_key=cache_key,
                )
            else:
                # Use cached voice embedding
                if voice_id not in _engine._voice_cache:
                    raise ValueError(f"Voice ID not found: {voice_id}")
                
                embedding = _engine._voice_cache[voice_id]
                start_time = time.perf_counter()
                
                speech_tokens, tokens_generated = _engine.generate_speech_tokens(text, embedding)
                audio = _engine.decode_audio(speech_tokens, embedding)
                
                generation_time = time.perf_counter() - start_time
                audio_duration = len(audio) / SAMPLE_RATE
                
                return TTSResult(
                    audio=audio,
                    sample_rate=SAMPLE_RATE,
                    generation_time=generation_time,
                    audio_duration=audio_duration,
                    rtf=generation_time / audio_duration if audio_duration > 0 else float('inf'),
                    tokens_generated=tokens_generated,
                )
        
        result = await loop.run_in_executor(_gpu_executor, _synthesize)
        
        logger.info(
            f"[{request_id}] Generated {result.audio_duration:.2f}s audio "
            f"in {result.generation_time:.2f}s (RTF={result.rtf:.3f})"
        )
        
        await update_metrics(result, success=True)
        
        # Return response
        if stream:
            # Streaming response
            def generate_audio():
                yield from _engine.stream_wav_chunks(result.audio)
            
            return StreamingResponse(
                generate_audio(),
                media_type="audio/wav",
                headers={
                    "X-Request-ID": request_id,
                    "X-Audio-Duration": str(result.audio_duration),
                    "X-Generation-Time": str(result.generation_time),
                    "X-RTF": str(result.rtf),
                    "X-Tokens-Generated": str(result.tokens_generated),
                },
            )
        else:
            # Full response with metadata
            audio_bytes = _engine.audio_to_wav_bytes(result.audio)
            return StreamingResponse(
                iter([audio_bytes]),
                media_type="audio/wav",
                headers={
                    "X-Request-ID": request_id,
                    "X-Audio-Duration": str(result.audio_duration),
                    "X-Generation-Time": str(result.generation_time),
                    "X-RTF": str(result.rtf),
                    "X-Tokens-Generated": str(result.tokens_generated),
                },
            )
    
    except ValueError as e:
        await update_metrics(success=False)
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        await update_metrics(success=False)
        logger.exception(f"[{request_id}] TTS failed: {e}")
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")
    
    finally:
        await track_concurrent(False)
        _semaphore.release()


@app.post("/tts/json", response_model=TTSResponse)
async def synthesize_speech_json(request: TTSRequest):
    """
    Synthesize speech and return metadata only (no audio).
    Useful for benchmarking without audio transfer overhead.
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    if not request.voice_id:
        raise HTTPException(
            status_code=400,
            detail="voice_id is required for JSON endpoint"
        )
    
    request_id = str(uuid.uuid4())[:8]
    
    async with asyncio.timeout(ServerConfig.REQUEST_TIMEOUT):
        await _semaphore.acquire()
    
    await track_concurrent(True)
    
    try:
        loop = asyncio.get_event_loop()
        
        def _synthesize():
            if request.voice_id not in _engine._voice_cache:
                raise ValueError(f"Voice ID not found: {request.voice_id}")
            
            embedding = _engine._voice_cache[request.voice_id]
            start_time = time.perf_counter()
            
            speech_tokens, tokens_generated = _engine.generate_speech_tokens(
                request.text, embedding
            )
            audio = _engine.decode_audio(speech_tokens, embedding)
            
            generation_time = time.perf_counter() - start_time
            audio_duration = len(audio) / SAMPLE_RATE
            
            return TTSResult(
                audio=audio,
                sample_rate=SAMPLE_RATE,
                generation_time=generation_time,
                audio_duration=audio_duration,
                rtf=generation_time / audio_duration if audio_duration > 0 else float('inf'),
                tokens_generated=tokens_generated,
            )
        
        result = await loop.run_in_executor(_gpu_executor, _synthesize)
        await update_metrics(result, success=True)
        
        return TTSResponse(
            request_id=request_id,
            audio_duration=result.audio_duration,
            generation_time=result.generation_time,
            rtf=result.rtf,
            tokens_generated=result.tokens_generated,
            sample_rate=result.sample_rate,
        )
    
    except ValueError as e:
        await update_metrics(success=False)
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        await update_metrics(success=False)
        logger.exception(f"[{request_id}] TTS failed: {e}")
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")
    
    finally:
        await track_concurrent(False)
        _semaphore.release()


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker for GPU sharing
        log_level="info",
    )
