# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn

# app = FastAPI(title="Baby Monitor AI Backend - Test Mode")

# # Configure CORS for React frontend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ==========================================
# # HEALTH CHECK ENDPOINT
# # ==========================================
# @app.get("/")
# def health_check():
#     return {
#         "status": "API is live and ready",
#         "mode": "test-mode (TensorFlow installation issue - using mock predictions)",
#         "port": 8000
#     }


# # ==========================================
# # AUDIO PREDICTION ENDPOINT (TEST)
# # ==========================================
# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     """
#     Test endpoint that returns mock predictions while we fix TensorFlow.
#     The return format matches what the frontend expects.
#     """
#     try:
#         print(f"📝 Received audio: {file.filename}")
        
#         # Mock prediction (in real implementation, this uses the TFLite model)
#         # Randomly return a crying or normal result for testing
#         import random
#         mock_probability = random.uniform(0.2, 0.95)
        
#         return {
#             "success": True,
#             "filename": file.filename,
#             "probability": mock_probability,
#             "is_crying": mock_probability > 0.80,
#             "label": "Crying" if mock_probability > 0.80 else "Normal",
#             "confidence": mock_probability,
#             "mode": "TEST - Returns Mock Data"
#         }

#     except Exception as e:
#         print(f"❌ Error: {e}")
#         return {
#             "success": False,
#             "error": str(e),
#             "label": "Error",
#             "confidence": 0.0
#         }


# # ==========================================
# # RUN THE SERVER
# # ==========================================
# if __name__ == "__main__":
#     print("\n" + "="*60)
#     print("🚀 Baby Monitor AI Backend - TEST MODE")
#     print("="*60)
#     print("⚠️  Running in test mode (mock predictions)")
#     print("📍 Server running at: http://localhost:8000")
#     print("📚 API docs at: http://localhost:8000/docs")
#     print("="*60 + "\n")
    
#     uvicorn.run(app, host="0.0.0.0", port=8000)





import numpy as np
import librosa
import pyaudio
import threading
import cv2
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --- Global State ---
# This keeps the latest results accessible to the FastAPI endpoint
latest_detection = {
    "label": "Initializing...",
    "confidence": 0.0,
    "is_crying": False,
    "status": "Listening"
}

# --- Import Logic for TFLite ---
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        tflite = None
        print("❌ Error: TFLite not found. Run: pip install tensorflow")

# --- Model Loading ---
MODEL_PATH = "baby_cry_v2_pro.tflite"
interpreter = None
input_details = None
output_details = None

if tflite and os.path.exists(MODEL_PATH):
    try:
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")

# --- Audio Config ---
RATE = 22050
CHUNK = 1024
DURATION = 5  # Analyze 5-second sliding windows

def process_audio(audio_data):
    """Processes audio buffer and returns AI prediction"""
    try:
        if interpreter is None:
            return "Model Error", 0

        # 1. Normalize and fix length
        audio_data = librosa.util.fix_length(audio_data, size=RATE * DURATION)
        if np.max(np.abs(audio_data)) > 0:
            audio_data = librosa.util.normalize(audio_data)
        
        # 2. Silence check (RMS)
        if np.sqrt(np.mean(audio_data**2)) < 0.01:
            return "Silence", 0.0

        # 3. Create Spectrogram (128x128)
        spec = librosa.feature.melspectrogram(y=audio_data, sr=RATE, n_mels=128)
        log_spec = librosa.power_to_db(spec, ref=np.max)
        resized = cv2.resize(log_spec, (128, 128))
        
        # 4. Prepare for Model (Batch, Height, Width, Channels)
        inp = np.expand_dims(resized, axis=(0, -1)).astype(np.float32)
        
        # 5. Inference
        interpreter.set_tensor(input_details[0]['index'], inp)
        interpreter.invoke()
        conf = float(interpreter.get_tensor(output_details[0]['index'])[0][0])
        
        # Thresholding
        label = "Baby Crying" if conf > 0.65 else "Normal"
        confidence_score = conf * 100 if label == "Baby Crying" else (1 - conf) * 100
        return label, confidence_score
    
    except Exception as e:
        print(f"⚠️ Inference Error: {e}")
        return "Error", 0

def mic_loop():
    """Background thread: Keeps the microphone ON and processes sound"""
    global latest_detection
    p = pyaudio.PyAudio()
    
    try:
        stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, 
                        input=True, frames_per_buffer=CHUNK)
        
        # Rolling buffer to hold 5 seconds of audio
        buffer = np.zeros(RATE * DURATION, dtype=np.float32)
        
        print("\n" + "="*40)
        print("🎙️  MIC ACTIVE: Monitoring Real-Time...")
        print("="*40 + "\n")

        while True:
            # Read small chunks to keep it responsive
            data = stream.read(CHUNK, exception_on_overflow=False)
            new_samples = np.frombuffer(data, dtype=np.float32)
            
            # Slide the buffer
            buffer = np.roll(buffer, -len(new_samples))
            buffer[-len(new_samples):] = new_samples
            
            # Process and update global state
            label, conf = process_audio(buffer)
            latest_detection = {
                "label": label,
                "confidence": round(conf, 2),
                "is_crying": label == "Baby Crying",
                "status": "Monitoring"
            }
            
            # Print Alert to Terminal
            if latest_detection["is_crying"]:
                print(f"🚨 ALERT: Baby Crying! ({conf:.1f}%)")

    except Exception as e:
        print(f"❌ Microphone Error: {e}")
    finally:
        p.terminate()

# --- FastAPI Backend ---
app = FastAPI(title="Baby Monitor AI - Real Time")

# Enable CORS for React/Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/status")
async def get_status():
    """Endpoint for frontend to poll real-time status"""
    return latest_detection

@app.get("/")
async def health():
    return {"status": "Backend is running", "mic_active": True}

if __name__ == "__main__":
    # Start Microphone thread as a Daemon (closes when main program stops)
    threading.Thread(target=mic_loop, daemon=True).start()
    
    # Start FastAPI server
    print(f"🚀 Server starting at http://localhost:8000")
    print(f"📖 Based on Reference [1]: Rupali P. et al., 2025")
    uvicorn.run(app, host="0.0.0.0", port=8000)