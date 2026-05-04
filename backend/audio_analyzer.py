import os
import torch
import numpy as np
from pydub import AudioSegment
import io
import re
import httpx
import scipy.io.wavfile as wavfile

class AudioAnalyzer:
    def __init__(self):
        # Load Silero VAD model
        import warnings
        warnings.filterwarnings("ignore")
        
        # Disable NNPACK to avoid "Unsupported hardware" spam on cloud CPUs
        try:
            torch.backends.nnpack.enabled = False
        except Exception:
            pass

        # We use torch.hub because it's the safest way to get the latest pre-trained model
        self.model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                              model='silero_vad',
                                              trust_repo=True, 
                                              force_reload=False,
                                              onnx=False)
        (self.get_speech_timestamps, _, self.read_audio, _, _) = self.utils
        self.sampling_rate = 16000 # Silero VAD expects 16kHz
        
        # Setup Groq API details
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_api_url = "https://api.groq.com/openai/v1/audio/transcriptions"
        self._COUNTER_Q_PATTERN = re.compile(
            r"\s*(,\s*)?(and\s+you|what\s+about\s+you|how\s+about\s+you|you\??)\s*\??\s*$",
            re.IGNORECASE,
        )

    def process_audio_blob(self, audio_base64):
        """
        Takes a base64 encoded audio blob (e.g. WebM/Opus from browser),
        converts it to 16kHz PCM, and returns speaking vs silence stats.
        """
        try:
            import base64
            header, encoded = audio_base64.split(",", 1) if "," in audio_base64 else (None, audio_base64)
            audio_data = base64.b64decode(encoded)
            
            # 1. Convert to PCM 16kHz Mono using pydub
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))
            audio_segment = audio_segment.set_frame_rate(self.sampling_rate).set_channels(1)
            
            # 2. Convert to float32 tensor as expected by Silero
            samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(samples)
            
            # 3. Get speech timestamps
            speech_timestamps = self.get_speech_timestamps(audio_tensor, self.model, sampling_rate=self.sampling_rate)
            
            # 4. Calculate stats
            total_duration_ms = len(audio_segment)
            total_speech_ms = 0
            
            for ts in speech_timestamps:
                # Timestamps are in samples, convert to ms
                start_ms = (ts['start'] / self.sampling_rate) * 1000
                end_ms = (ts['end'] / self.sampling_rate) * 1000
                total_speech_ms += (end_ms - start_ms)
            
            total_silence_ms = total_duration_ms - total_speech_ms
            
            # Calculate trailing silence
            trailing_silence_ms = total_duration_ms
            if len(speech_timestamps) > 0:
                last_end_ms = (speech_timestamps[-1]['end'] / self.sampling_rate) * 1000
                trailing_silence_ms = total_duration_ms - last_end_ms

            # Return stats AND the raw samples for buffering
            return {
                "speech_ms": total_speech_ms,
                "silence_ms": total_silence_ms,
                "trailing_silence_ms": max(0, trailing_silence_ms),
                "duration_ms": total_duration_ms,
                "samples": samples
            }
        except Exception as e:
            print(f"Audio Analysis Error: {e}")
            return None

    def _clean_user_input(self, text: str) -> str:
        cleaned = self._COUNTER_Q_PATTERN.sub("", text).strip()
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned or text

    async def transcribe_buffer(self, samples):
        if len(samples) == 0:
            return ""

        try:
            samples_int16 = (samples * 32767.0).astype(np.int16)
            wav_io = io.BytesIO()
            wavfile.write(wav_io, self.sampling_rate, samples_int16)
            wav_bytes = wav_io.getvalue()

            async with httpx.AsyncClient(timeout=12.0) as client:
                res = await client.post(
                    self.groq_api_url,
                    files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                    data={
                        "model": "whisper-large-v3-turbo",
                        "language": "en",
                        "temperature": "0.0"
                    },
                    headers={"Authorization": f"Bearer {self.groq_api_key}"},
                )
            
            if res.status_code == 200:
                text = res.json().get("text", "").strip()
                
                _HALLUCINATIONS = {
                    "Thank you.", "Thank you", "Thanks for watching.",
                    "Thank you very much.", "Thank you for watching.",
                    "Bye.", "Thanks.", "Thank you!", "What a Christian!",
                    "What a Christian", "you", "You.", "No.", "No",
                    "So.", "so", "Yeah.", "yeah"
                }
                if text in _HALLUCINATIONS or len(text) < 3:
                    print(f"🎤 [STT] Dropping hallucination: '{text}'")
                    return ""
                
                return self._clean_user_input(text)
            else:
                print(f"Groq API Error {res.status_code}: {res.text[:200]}")
                return ""
        except Exception as e:
            print(f"Groq STT Exception: {e}")
            return ""

    async def transcribe_webm(self, audio_base64: str) -> str:
        """Transcribes raw WebM base64 blob directly using Groq."""
        try:
            import base64
            header, encoded = audio_base64.split(",", 1) if "," in audio_base64 else (None, audio_base64)
            audio_bytes = base64.b64decode(encoded)

            async with httpx.AsyncClient(timeout=12.0) as client:
                res = await client.post(
                    self.groq_api_url,
                    files={"file": ("audio.webm", audio_bytes, "audio/webm")},
                    data={
                        "model": "whisper-large-v3-turbo",
                        "language": "en",
                        "temperature": "0.0"
                    },
                    headers={"Authorization": f"Bearer {self.groq_api_key}"},
                )
            
            if res.status_code == 200:
                text = res.json().get("text", "").strip()
                _HALLUCINATIONS = {
                    "Thank you.", "Thank you", "Thanks for watching.",
                    "Thank you very much.", "Thank you for watching.",
                    "Bye.", "Thanks.", "Thank you!", "What a Christian!",
                    "What a Christian", "you", "You.", "No.", "No",
                    "So.", "so", "Yeah.", "yeah"
                }
                if text in _HALLUCINATIONS or len(text) < 3:
                    return ""
                return self._clean_user_input(text)
            return ""
        except Exception as e:
            print(f"Direct WebM Transcribe Error: {e}")
            return ""
