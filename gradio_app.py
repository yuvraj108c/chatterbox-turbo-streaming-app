"""
Gradio interface for Chatterbox TTS API
Automatically starts the API server in the background
"""

import gradio as gr
import requests
import tempfile
import subprocess
import time
import sys
import os
import signal
import atexit

# API configuration
API_URL = "http://localhost:8000"
API_PROCESS = None


def start_api_server():
    """Start the FastAPI server in the background."""
    global API_PROCESS
    
    try:
        print("🚀 Starting TTS API server...")
        print("=" * 60)
        
        # Start the server as a subprocess without capturing output
        # This allows logs to be printed to console
        API_PROCESS = subprocess.Popen(
            [sys.executable, "server.py"],
            stdout=None,  # Don't capture stdout - let it print
            stderr=None,  # Don't capture stderr - let it print
        )
        
        # Wait for server to be ready (max 60 seconds for model loading)
        print("⏳ Waiting for TTS server to initialize (this may take a minute)...")
        for i in range(60):
            time.sleep(1)
            try:
                response = requests.get(f"{API_URL}/health", timeout=2)
                if response.status_code == 200:
                    print("=" * 60)
                    print("✅ TTS API server is ready!")
                    return True
            except:
                if i % 5 == 0:  # Print every 5 seconds
                    print(f"⏳ Still waiting... ({i+1}/60)")
                continue
        
        print("=" * 60)
        print("❌ Server failed to start within 60 seconds")
        return False
        
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        return False


def stop_api_server():
    """Stop the FastAPI server."""
    global API_PROCESS
    
    if API_PROCESS:
        print("🛑 Stopping TTS API server...")
        API_PROCESS.terminate()
        try:
            API_PROCESS.wait(timeout=5)
        except subprocess.TimeoutExpired:
            API_PROCESS.kill()
        print("✅ TTS API server stopped")


# Register cleanup handler
atexit.register(stop_api_server)


def synthesize_speech(text, voice_audio):
    """
    Call the TTS API to synthesize speech.
    
    Args:
        text: Text to synthesize
        voice_audio: Path to reference voice audio file
    
    Returns:
        Tuple of (audio_path, info_text)
    """
    if not text or not text.strip():
        return None, "⚠️ Please enter some text to synthesize."
    
    if not voice_audio:
        return None, "⚠️ Please upload a reference voice audio file."
    
    try:
        # Prepare the request
        files = {
            'voice_file': ('voice.wav', open(voice_audio, 'rb'), 'audio/wav')
        }
        data = {
            'text': text,
            'stream': False
        }
        
        # Call the API
        response = requests.post(
            f"{API_URL}/tts",
            files=files,
            data=data,
            timeout=120
        )
        
        if response.status_code != 200:
            return None, f"❌ Error: {response.status_code} - {response.text}"
        
        # Get metadata from headers
        audio_duration = response.headers.get('X-Audio-Duration', 'N/A')
        generation_time = response.headers.get('X-Generation-Time', 'N/A')
        rtf = response.headers.get('X-RTF', 'N/A')
        tokens = response.headers.get('X-Tokens-Generated', 'N/A')
        
        # Save the audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(response.content)
            audio_path = tmp_file.name
        
        # Create info text
        info = f"""
✅ **Synthesis Complete**

- **Audio Duration:** {audio_duration}s
- **Generation Time:** {generation_time}s
- **Real-Time Factor (RTF):** {rtf}
- **Tokens Generated:** {tokens}
        """
        
        return audio_path, info
    
    except requests.exceptions.RequestException as e:
        return None, f"❌ API Error: {str(e)}\n\nMake sure the TTS server is running at {API_URL}"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def check_health():
    """Check if the API server is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return f"🟢 **Server Status:** {data['status']}\n**GPU:** {data['gpu_provider']}\n**Model:** {data['model_dtype']}\n**Cached Voices:** {data['voice_cache_size']}"
        else:
            return f"🔴 Server returned status code: {response.status_code}"
    except:
        return f"🔴 Server not reachable at {API_URL}"


# Create Gradio interface
with gr.Blocks(title="Chatterbox TTS", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎙️ Chatterbox TTS
        
        Voice cloning text-to-speech using your reference audio.
        """
    )
    
    # Server status
    with gr.Row():
        status_text = gr.Markdown(
            value="⏳ Starting server...",
            label="Server Status"
        )
        refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
    
    refresh_btn.click(fn=check_health, outputs=status_text)
    
    gr.Markdown("---")
    
    # Main interface
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Text to Synthesize",
                placeholder="Enter the text you want to convert to speech...",
                lines=5,
                max_lines=10
            )
            
            voice_input = gr.Audio(
                label="Reference Voice Audio",
                type="filepath",
                sources=["upload", "microphone"]
            )
            
            synthesize_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg")
        
        with gr.Column():
            audio_output = gr.Audio(
                label="Generated Speech",
                type="filepath"
            )
            
            info_output = gr.Markdown(value="")
    
    # Connect the button
    synthesize_btn.click(
        fn=synthesize_speech,
        inputs=[text_input, voice_input],
        outputs=[audio_output, info_output]
    )
    
    # Examples
    gr.Markdown("---")
    gr.Markdown("### 📝 Tips")
    gr.Markdown(
        """
        - Upload a clear reference voice audio (10-30 seconds recommended)
        - The generated voice will mimic the style and tone of your reference audio
        - Text can be up to 5000 characters
        - Supported audio formats: WAV, MP3, FLAC, etc.
        """
    )
    
    # Update status on load
    demo.load(fn=check_health, outputs=status_text)


if __name__ == "__main__":
    # Start the API server first
    if start_api_server():
        # Launch Gradio interface
        try:
            demo.launch(
                server_name="0.0.0.0",
                server_port=7860,
                share=False
            )
        except KeyboardInterrupt:
            print("\n👋 Shutting down...")
        finally:
            stop_api_server()
    else:
        print("❌ Failed to start API server. Exiting.")
        sys.exit(1)