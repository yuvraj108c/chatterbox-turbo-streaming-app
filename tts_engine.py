"""
TTSEngine: GPU-optimized Chatterbox TTS inference engine with ONNX Runtime.
Optimized for L40S GPU with FP16 support and voice embedding caching.
"""

import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import numpy as np
import librosa
import io
import wave
import time
from typing import Optional, Tuple, Dict, Any, Generator
from dataclasses import dataclass
from functools import lru_cache
import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model constants
MODEL_ID = "ResembleAI/chatterbox-turbo-ONNX"
SAMPLE_RATE = 24000
START_SPEECH_TOKEN = 6561
STOP_SPEECH_TOKEN = 6562
SILENCE_TOKEN = 4299
NUM_KV_HEADS = 16
HEAD_DIM = 64


@dataclass
class TTSConfig:
    """Configuration for TTS engine."""
    model_dtype: str = "fp16"  # fp32, fp16, q8, q4, q4f16
    max_new_tokens: int = 1024
    repetition_penalty: float = 1.2
    gpu_device_id: int = 0
    gpu_mem_limit_gb: int = 40  # L40S has 48GB, leave headroom
    voice_cache_size: int = 100
    apply_watermark: bool = False


@dataclass
class VoiceEmbedding:
    """Cached voice embeddings for fast reuse."""
    cond_emb: np.ndarray
    prompt_token: np.ndarray
    speaker_embeddings: np.ndarray
    speaker_features: np.ndarray


@dataclass
class TTSResult:
    """Result of TTS generation."""
    audio: np.ndarray
    sample_rate: int
    generation_time: float
    audio_duration: float
    rtf: float  # Real-Time Factor
    tokens_generated: int


class RepetitionPenaltyLogitsProcessor:
    """Apply repetition penalty to logits during generation."""
    
    def __init__(self, penalty: float):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` must be a strictly positive float, but is {penalty}")
        self.penalty = penalty

    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        score = np.take_along_axis(scores, input_ids, axis=1)
        score = np.where(score < 0, score * self.penalty, score / self.penalty)
        scores_processed = scores.copy()
        np.put_along_axis(scores_processed, input_ids, score, axis=1)
        return scores_processed


class TTSEngine:
    """
    GPU-optimized TTS Engine using Chatterbox Turbo ONNX models.
    
    Features:
    - CUDA execution with L40S optimization
    - FP16 inference for better throughput
    - Voice embedding caching
    - Thread-safe inference
    """
    
    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig()
        self._lock = threading.Lock()
        self._gpu_lock = threading.Lock()  # Serialize GPU inference to prevent OOM
        self._voice_cache: Dict[str, VoiceEmbedding] = {}
        
        logger.info(f"Initializing TTSEngine with dtype={self.config.model_dtype}")
        
        # Download and load models
        self._download_models()
        self._create_sessions()
        self._load_tokenizer()
        
        logger.info("TTSEngine initialized successfully")
    
    def _download_models(self):
        """Download ONNX models from HuggingFace Hub."""
        logger.info("Downloading ONNX models...")
        dtype = self.config.model_dtype
        
        self.model_paths = {
            "speech_encoder": self._download_model("speech_encoder", dtype),
            "embed_tokens": self._download_model("embed_tokens", dtype),
            "language_model": self._download_model("language_model", dtype),
            "conditional_decoder": self._download_model("conditional_decoder", dtype),
        }
        logger.info("Models downloaded")
    
    def _download_model(self, name: str, dtype: str = "fp32") -> str:
        """Download a single model file."""
        filename = f"{name}{'' if dtype == 'fp32' else '_quantized' if dtype == 'q8' else f'_{dtype}'}.onnx"
        graph = hf_hub_download(MODEL_ID, subfolder="onnx", filename=filename)
        hf_hub_download(MODEL_ID, subfolder="onnx", filename=f"{filename}_data")
        return graph
    
    def _create_sessions(self):
        """Create ONNX Runtime sessions with GPU providers."""
        logger.info("Creating ONNX sessions with CUDA provider...")
        
        # Configure CUDA provider for L40S with optimized conv settings
        # Don't set gpu_mem_limit - let CUDA manage memory dynamically
        # This prevents BFC arena fragmentation issues
        cuda_options = {
            'device_id': self.config.gpu_device_id,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'cudnn_conv_algo_search': 'HEURISTIC',  # Better for FP16 conv fallbacks
            'cudnn_conv_use_max_workspace': '1',     # Allow more workspace for conv ops
            'cudnn_conv1d_pad_to_nc1d': '1',         # Optimize 1D convolutions
            'do_copy_in_default_stream': True,
        }
        
        # Use CUDA provider (TensorRT would be faster but requires installation)
        providers = [
            ('CUDAExecutionProvider', cuda_options),
            'CPUExecutionProvider'
        ]
        
        # Session options for optimization
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.enable_mem_pattern = True
        session_options.enable_mem_reuse = True
        session_options.intra_op_num_threads = 4
        session_options.inter_op_num_threads = 2
        
        # Create sessions
        self.speech_encoder_session = ort.InferenceSession(
            self.model_paths["speech_encoder"],
            sess_options=session_options,
            providers=providers
        )
        self.embed_tokens_session = ort.InferenceSession(
            self.model_paths["embed_tokens"],
            sess_options=session_options,
            providers=providers
        )
        self.language_model_session = ort.InferenceSession(
            self.model_paths["language_model"],
            sess_options=session_options,
            providers=providers
        )
        self.cond_decoder_session = ort.InferenceSession(
            self.model_paths["conditional_decoder"],
            sess_options=session_options,
            providers=providers
        )
        
        # Verify GPU is being used
        for name, session in [
            ("speech_encoder", self.speech_encoder_session),
            ("language_model", self.language_model_session),
        ]:
            provider = session.get_providers()[0]
            logger.info(f"{name} using provider: {provider}")
    
    def _load_tokenizer(self):
        """Load HuggingFace tokenizer."""
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    def encode_voice(self, audio_path: str) -> VoiceEmbedding:
        """
        Encode a reference voice audio into embeddings.
        Results are cached for reuse.
        
        Args:
            audio_path: Path to reference voice WAV file
            
        Returns:
            VoiceEmbedding containing all speaker embeddings
        """
        # Check cache
        if audio_path in self._voice_cache:
            logger.debug(f"Using cached voice embedding for {audio_path}")
            return self._voice_cache[audio_path]
        
        logger.info(f"Encoding voice from {audio_path}")
        
        # Load and preprocess audio
        audio_values, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
        audio_values = audio_values[np.newaxis, :].astype(np.float32)
        
        # Run speech encoder with GPU lock
        with self._gpu_lock:
            cond_emb, prompt_token, speaker_embeddings, speaker_features = \
                self.speech_encoder_session.run(None, {"audio_values": audio_values})
        
        embedding = VoiceEmbedding(
            cond_emb=cond_emb,
            prompt_token=prompt_token,
            speaker_embeddings=speaker_embeddings,
            speaker_features=speaker_features,
        )
        
        # Cache with size limit
        if len(self._voice_cache) >= self.config.voice_cache_size:
            # Remove oldest entry
            oldest = next(iter(self._voice_cache))
            del self._voice_cache[oldest]
        
        self._voice_cache[audio_path] = embedding
        return embedding
    
    def encode_voice_from_bytes(self, audio_bytes: bytes, cache_key: Optional[str] = None) -> VoiceEmbedding:
        """
        Encode voice from audio bytes (for API uploads).
        
        Args:
            audio_bytes: Raw audio file bytes
            cache_key: Optional key for caching
            
        Returns:
            VoiceEmbedding
        """
        if cache_key and cache_key in self._voice_cache:
            return self._voice_cache[cache_key]
        
        # Load from bytes
        audio_values, _ = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE)
        audio_values = audio_values[np.newaxis, :].astype(np.float32)
        
        # Run speech encoder with GPU lock
        with self._gpu_lock:
            cond_emb, prompt_token, speaker_embeddings, speaker_features = \
                self.speech_encoder_session.run(None, {"audio_values": audio_values})
        
        embedding = VoiceEmbedding(
            cond_emb=cond_emb,
            prompt_token=prompt_token,
            speaker_embeddings=speaker_embeddings,
            speaker_features=speaker_features,
        )
        
        if cache_key:
            if len(self._voice_cache) >= self.config.voice_cache_size:
                oldest = next(iter(self._voice_cache))
                del self._voice_cache[oldest]
            self._voice_cache[cache_key] = embedding
        
        return embedding
    
    def generate_speech_tokens(
        self,
        text: str,
        voice_embedding: VoiceEmbedding,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech tokens from text using autoregressive LM.
        
        Args:
            text: Input text to synthesize
            voice_embedding: Pre-encoded voice embedding
            
        Returns:
            Tuple of (speech_tokens, tokens_generated)
        """
        # Acquire GPU lock to prevent concurrent GPU memory allocation issues
        with self._gpu_lock:
            return self._generate_speech_tokens_impl(text, voice_embedding)
    
    def _generate_speech_tokens_impl(
        self,
        text: str,
        voice_embedding: VoiceEmbedding,
    ) -> Tuple[np.ndarray, int]:
        """Internal implementation of speech token generation."""
        # Tokenize text
        input_ids = self.tokenizer(text, return_tensors="np")["input_ids"].astype(np.int64)
        
        # Initialize
        repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(
            penalty=self.config.repetition_penalty
        )
        generate_tokens = np.array([[START_SPEECH_TOKEN]], dtype=np.int64)
        
        # Get KV cache dtype from model
        kv_dtype = np.float32
        for inp in self.language_model_session.get_inputs():
            if "past_key_values" in inp.name:
                kv_dtype = np.float16 if inp.type == 'tensor(float16)' else np.float32
                break
        
        past_key_values = None
        attention_mask = None
        position_ids = None
        
        for i in range(self.config.max_new_tokens):
            inputs_embeds = self.embed_tokens_session.run(None, {"input_ids": input_ids})[0]
            
            if i == 0:
                # First iteration: concatenate voice conditioning
                inputs_embeds = np.concatenate(
                    (voice_embedding.cond_emb, inputs_embeds), axis=1
                )
                
                batch_size, seq_len, _ = inputs_embeds.shape
                past_key_values = {
                    inp.name: np.zeros([batch_size, NUM_KV_HEADS, 0, HEAD_DIM], dtype=kv_dtype)
                    for inp in self.language_model_session.get_inputs()
                    if "past_key_values" in inp.name
                }
                attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
                position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1).repeat(batch_size, axis=0)
            
            # Run language model
            logits, *present_key_values = self.language_model_session.run(
                None,
                dict(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **past_key_values,
                )
            )
            
            # Sample next token
            logits = logits[:, -1, :]
            next_token_logits = repetition_penalty_processor(generate_tokens, logits)
            input_ids = np.argmax(next_token_logits, axis=-1, keepdims=True).astype(np.int64)
            generate_tokens = np.concatenate((generate_tokens, input_ids), axis=-1)
            
            # Check for stop token
            if (input_ids.flatten() == STOP_SPEECH_TOKEN).all():
                break
            
            # Update for next iteration
            batch_size = attention_mask.shape[0]
            attention_mask = np.concatenate(
                [attention_mask, np.ones((batch_size, 1), dtype=np.int64)], axis=1
            )
            position_ids = position_ids[:, -1:] + 1
            for j, key in enumerate(past_key_values):
                past_key_values[key] = present_key_values[j]
        
        tokens_generated = generate_tokens.shape[1] - 1  # Exclude start token
        return generate_tokens, tokens_generated
    
    def decode_audio(
        self,
        speech_tokens: np.ndarray,
        voice_embedding: VoiceEmbedding,
    ) -> np.ndarray:
        """
        Decode speech tokens to audio waveform.
        
        Args:
            speech_tokens: Generated speech tokens
            voice_embedding: Voice embedding for conditioning
            
        Returns:
            Audio waveform as numpy array
        """
        # Acquire GPU lock for decoder inference
        with self._gpu_lock:
            return self._decode_audio_impl(speech_tokens, voice_embedding)
    
    def _decode_audio_impl(
        self,
        speech_tokens: np.ndarray,
        voice_embedding: VoiceEmbedding,
    ) -> np.ndarray:
        """Internal implementation of audio decoding."""
        # Prepare tokens: remove start/stop, add prompt and silence
        tokens = speech_tokens[:, 1:-1]  # Remove START and STOP tokens
        silence_tokens = np.full((tokens.shape[0], 3), SILENCE_TOKEN, dtype=np.int64)
        tokens = np.concatenate([voice_embedding.prompt_token, tokens, silence_tokens], axis=1)
        
        # Run decoder
        wav = self.cond_decoder_session.run(
            None,
            dict(
                speech_tokens=tokens,
                speaker_embeddings=voice_embedding.speaker_embeddings,
                speaker_features=voice_embedding.speaker_features,
            )
        )[0].squeeze(axis=0)
        
        # Optional watermark
        if self.config.apply_watermark:
            try:
                import perth
                watermarker = perth.PerthImplicitWatermarker()
                wav = watermarker.apply_watermark(wav, sample_rate=SAMPLE_RATE)
            except ImportError:
                logger.warning("perth not installed, skipping watermark")
        
        return wav
    
    def synthesize(
        self,
        text: str,
        voice_path: Optional[str] = None,
        voice_bytes: Optional[bytes] = None,
        voice_cache_key: Optional[str] = None,
    ) -> TTSResult:
        """
        Full TTS pipeline: text -> audio.
        
        Args:
            text: Text to synthesize
            voice_path: Path to reference voice file
            voice_bytes: Raw voice audio bytes (alternative to path)
            voice_cache_key: Cache key for voice embedding
            
        Returns:
            TTSResult with audio and metrics
        """
        start_time = time.perf_counter()
        
        # Encode voice
        if voice_path:
            voice_embedding = self.encode_voice(voice_path)
        elif voice_bytes:
            voice_embedding = self.encode_voice_from_bytes(voice_bytes, voice_cache_key)
        else:
            raise ValueError("Either voice_path or voice_bytes must be provided")
        
        # Generate speech tokens
        speech_tokens, tokens_generated = self.generate_speech_tokens(text, voice_embedding)
        
        # Decode to audio
        audio = self.decode_audio(speech_tokens, voice_embedding)
        
        # Calculate metrics
        generation_time = time.perf_counter() - start_time
        audio_duration = len(audio) / SAMPLE_RATE
        rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')
        
        return TTSResult(
            audio=audio,
            sample_rate=SAMPLE_RATE,
            generation_time=generation_time,
            audio_duration=audio_duration,
            rtf=rtf,
            tokens_generated=tokens_generated,
        )
    
    def audio_to_wav_bytes(self, audio: np.ndarray) -> bytes:
        """Convert audio numpy array to WAV bytes."""
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(SAMPLE_RATE)
            # Convert float32 to int16
            audio_int16 = (audio * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())
        return buffer.getvalue()
    
    def stream_wav_chunks(
        self,
        audio: np.ndarray,
        chunk_size: int = 4096,
    ) -> Generator[bytes, None, None]:
        """
        Stream audio as WAV chunks for HTTP streaming.
        
        Args:
            audio: Audio waveform
            chunk_size: Samples per chunk
            
        Yields:
            WAV file chunks (header + data chunks)
        """
        # First yield: WAV header
        buffer = io.BytesIO()
        audio_int16 = (audio * 32767).astype(np.int16)
        
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(audio_int16.tobytes())
        
        # Yield in chunks
        buffer.seek(0)
        while True:
            chunk = buffer.read(chunk_size)
            if not chunk:
                break
            yield chunk
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "model_dtype": self.config.model_dtype,
            "voice_cache_size": len(self._voice_cache),
            "voice_cache_limit": self.config.voice_cache_size,
            "providers": self.language_model_session.get_providers(),
        }


# Singleton instance for the server
_engine_instance: Optional[TTSEngine] = None
_engine_lock = threading.Lock()


def get_engine(config: Optional[TTSConfig] = None) -> TTSEngine:
    """Get or create the singleton TTSEngine instance."""
    global _engine_instance
    
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                _engine_instance = TTSEngine(config)
    
    return _engine_instance
