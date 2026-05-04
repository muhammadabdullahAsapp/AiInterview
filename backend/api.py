from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import uvicorn
import base64
import numpy as np
import time
import anyio
from typing import Optional
import socketio
from face_analyzer import FaceAnalyzer
from audio_analyzer import AudioAnalyzer
import gc
import os
import httpx
import asyncio
from dotenv import load_dotenv

load_dotenv()

# 1. Initialize Socket.io Server with verbose logging for debugging
sio = socketio.AsyncServer(
    async_mode='asgi', 
    cors_allowed_origins='*', # Hardcoded * for now to bypass 403
    logger=True, 
    engineio_logger=True,
    always_connect=True
)
app = FastAPI()

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Move socket app wrapping here to be cleaner
socket_app = socketio.ASGIApp(sio, app)

# Global variables
analyzer = None
audio_analyzer = None

class FrameData(BaseModel):
    image: str # Base64 encoded image string
    session_id: str = "default" # Unique ID per user tab
    room_id: str = None # Optional room for relay

class AudioData(BaseModel):
    audio: str # Base64 encoded audio blob
    session_id: str = "default"
    room_id: str = None

class SessionClearRequest(BaseModel):
    session_id: str

@app.on_event("startup")
async def startup_event():
    global analyzer, audio_analyzer
    # Initialize Analyzers
    analyzer = FaceAnalyzer()
    audio_analyzer = AudioAnalyzer()
    print("System Started - Waiting for frames and audio...")

@app.on_event("shutdown")
async def shutdown_event():
    global analyzer
    if analyzer:
        analyzer.stop()
    print("System Shutdown")

@app.post("/end_session")
async def end_session(data: SessionClearRequest):
    global analyzer
    if analyzer:
        with analyzer.lock:
            if data.session_id in analyzer.sessions:
                session_data = analyzer.sessions[data.session_id]
                full_transcript = session_data.get('full_transcript', '').strip()
                
                # Trigger Webhook
                webhook_url = os.getenv("WEBHOOK_URL")
                if webhook_url and webhook_url != "https://webhook.site/replace-with-your-id":
                    print(f"🚀 Sending transcript to Webhook: {webhook_url}")
                    asyncio.create_task(send_webhook(webhook_url, data.session_id, full_transcript))
                else:
                    print(f"⚠️ WEBHOOK_URL not configured. Transcript would be:\n{full_transcript}")

                del analyzer.sessions[data.session_id]
                gc.collect() # Force garbage collection to free RAM
                print(f"Session {data.session_id} deleted from RAM and GC triggered.")
                return {"success": True, "message": "Session cleared"}
    return {"success": False, "message": "Session not found"}

async def send_webhook(url: str, session_id: str, transcript: str):
    try:
        async with httpx.AsyncClient() as client:
            payload = {
                "session_id": session_id,
                "transcript": transcript,
                "timestamp": time.time()
            }
            response = await client.post(url, json=payload)
            print(f"✅ Webhook Success ({response.status_code}): {response.text[:100]}")
    except Exception as e:
        print(f"❌ Webhook Failed: {e}")

@app.post("/analyze")
async def analyze_frame(data: FrameData):
    global analyzer
    if analyzer is None:
        raise HTTPException(status_code=500, detail="Analyzer not initialized")

    try:
        # 1. Decode Base64 string to OpenCV Image
        header, encoded = data.image.split(",", 1) if "," in data.image else (None, data.image)
        try:
            nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as decode_err:
            print(f"Base64 Decoding Error: {decode_err}")
            raise ValueError(f"Invalid image format: {decode_err}")

        if frame is None:
            print("Decoding Error: frame is None")
            raise ValueError("Invalid image data: cv2.imdecode failed")

        # 2. Analyze with session isolation (OFFLOAD TO THREAD)
        # Using timeout to prevent thread pile-up if processing hangs
        results = await anyio.to_thread.run_sync(analyzer.analyze_frame_sync, frame, data.session_id)
        
        if results and len(results) > 0:
            face = results[0]
            
            # Unified Confidence Score (0-100)
            g_score = face.get('gaze_score', 0)
            s_score = face.get('stability_score', 0)
            confidence = (g_score * 50) + (s_score * 50)
            
            results_payload = {
                "detected": True,
                "dominant_emotion": face.get('dominant_emotion'),
                "emotions": face.get('emotions'),
                "gaze_score": round(g_score, 2),
                "stability_score": round(s_score, 2),
                "confidence_score": round(confidence, 1)
            }

            # Relay to room if in an interview
            if data.room_id:
                try:
                    await sio.emit('ai_results', results_payload, room=data.room_id)
                except Exception as sio_err:
                    print(f"Socket.io Relay Error: {sio_err}")
            
            return results_payload
        
        return {"detected": False}

    except Exception as e:
        import traceback
        print(f"CRITICAL Error processing frame: {e}")
        traceback.print_exc()
        return {"detected": False, "error": str(e)}

@app.post("/analyze_audio")
async def analyze_audio(data: AudioData):
    global analyzer, audio_analyzer
    if analyzer is None or audio_analyzer is None:
        raise HTTPException(status_code=500, detail="Analyzers not initialized")

    try:
        # 1. Process audio blob (OFFLOAD TO THREAD)
        stats = await anyio.to_thread.run_sync(audio_analyzer.process_audio_blob, data.audio)
        
        if stats:
            # 2. Update session state
            with analyzer.lock:
                # Ensure session and critical keys exist (robust against backend restarts)
                if data.session_id not in analyzer.sessions:
                    analyzer.sessions[data.session_id] = {
                        "emotions": {},
                        "audio_stats": {
                            "speech_ms": 0, 
                            "silence_ms": 0, 
                            "current_silence_ms": 0
                        },
                        "last_head_pos": (0.5, 0.5),
                        "stability_history": [1.0] * 10,
                        "last_seen": time.time(),
                        "full_transcript": ""
                    }
                
                session = analyzer.sessions[data.session_id]
                session['last_seen'] = time.time()
                
                if 'audio_stats' not in session:
                    session['audio_stats'] = {
                        "speech_ms": 0, 
                        "silence_ms": 0, 
                        "current_silence_ms": 0
                    }
                
                if 'audio_buffer' not in session:
                    session['audio_buffer'] = np.array([], dtype=np.float32)
                
                if 'full_transcript' not in session:
                    session['full_transcript'] = ""
                
                # Robustness for face state (if session was created by audio)
                if 'stability_history' not in session:
                    session['stability_history'] = [1.0] * 10
                    session['last_head_pos'] = (0.5, 0.5)
                
                s_stats = session['audio_stats']

                # Logic: If user spoke a significant amount, reset streak to the silence AFTER speech.
                # If blob was mostly silent/noise, continue the existing streak.
                SPEECH_THRESHOLD_MS = 100 # Ignore sounds shorter than 100ms as noise
                
                if stats.get('speech_ms', 0) > SPEECH_THRESHOLD_MS:
                    # Significant speech detected - streak is just the silence at the tail end
                    s_stats['current_silence_ms'] = stats.get('trailing_silence_ms', 0)
                else:
                    # Mostly silent blob - add the entire blob's silence to the streak
                    s_stats['current_silence_ms'] += stats.get('silence_ms', 0)

                s_stats['speech_ms'] = s_stats.get('speech_ms', 0) + stats.get('speech_ms', 0)
                s_stats['silence_ms'] = s_stats.get('silence_ms', 0) + stats.get('silence_ms', 0)
                
                # Determine Vocal Status based on current streak
                streak = s_stats['current_silence_ms']
                status = "fluent"
                if streak > 10000:
                    status = "freeze"
                elif streak > 5000:
                    status = "stalling"
                elif streak > 2000:
                    status = "thinking"

                # Calculate cumulative fluency
                total_time = s_stats.get('speech_ms', 0) + s_stats.get('silence_ms', 0)
                fluency = (s_stats.get('speech_ms', 0) / total_time * 100) if total_time > 0 else 100
                
                # Buffer Management & Transcription
                # Because the frontend uses hark VAD, every payload is a perfectly bounded turn!
                transcription = await audio_analyzer.transcribe_webm(data.audio)
                is_final = True
                
                if transcription:
                    session['full_transcript'] += transcription + " "
                
                # Cleanup old buffer memory
                if 'audio_buffer' in session:
                    session['audio_buffer'] = np.array([], dtype=np.float32)
                
                results_payload = {
                    "success": True,
                    "fluency": round(fluency, 2),
                    "is_speaking": stats.get('speech_ms', 0) > 0,
                    "vocal_status": status,
                    "silence_streak": round(streak / 1000, 1),
                    "transcription": transcription,
                    "is_final": is_final
                }

                # Relay to room if in an interview
                if data.room_id:
                    await sio.emit('vocal_results', results_payload, room=data.room_id)
                
                return results_payload
        
        return {"success": False, "error": "Could not analyze audio"}

    except Exception as e:
        print(f"Error processing audio: {e}")
        return {"success": False, "error": str(e)}

@app.get("/status")
async def get_status():
    return {"status": "online", "model": "DeepFace (MediaPipe backend)"}

# --- Socket.io Handlers ---


@sio.event
async def connect(sid, environ):
    print(f"Client Connected: {sid}")

@sio.on('join_room')
async def handle_join_room(sid, data):
    room = data.get('room')
    if room:
        await sio.enter_room(sid, room)
        print(f"Client {sid} joined room: {room}")
        # Tell others in the room a new user has joined
        await sio.emit('user_joined', {"sid": sid}, room=room, skip_sid=sid)
        await sio.emit('room_joined', {"room": room}, room=room)

@sio.event
async def leave_room(sid, data):
    room = data.get('room')
    if room:
        await sio.leave_room(sid, room)
        print(f"Client {sid} left room: {room}")

@sio.event
async def offer(sid, data):
    # Relay RTC Offer to others in the room
    room = data.get('room')
    print(f"Relaying Offer for room: {room}")
    await sio.emit('offer', data, room=room, skip_sid=sid)

@sio.event
async def answer(sid, data):
    # Relay RTC Answer to others in the room
    room = data.get('room')
    print(f"Relaying Answer for room: {room}")
    await sio.emit('answer', data, room=room, skip_sid=sid)

@sio.event
async def ice_candidate(sid, data):
    # Relay ICE Candidate to others in the room
    room = data.get('room')
    await sio.emit('ice_candidate', data, room=room, skip_sid=sid)

@sio.event
async def terminate_room(sid, data):
    room = data.get('room')
    print(f"Room Terminated by Interviewer: {room}")
    await sio.emit('room_terminated', {"room": room}, room=room)

@sio.event
async def disconnect(sid):
    print(f"Client Disconnected: {sid}")

if __name__ == "__main__":
    uvicorn.run(socket_app, host="0.0.0.0", port=8000)
