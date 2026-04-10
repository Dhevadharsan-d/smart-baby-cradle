# import os
# import tempfile
# import librosa
# import numpy as np
# import tensorflow as tf
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import uvicorn
# import traceback
# from pydub import AudioSegment
# import io
# import os


# # This forces pydub to use exactly this location, bypassing Windows Path issues.
# # AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
# # AudioSegment.ffprobe   = r"C:\ffmpeg\bin\ffprobe.exe"

# app = FastAPI(title="Baby Monitor AI Backend")

# # Configure CORS for React
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins (React frontend, IoT devices)
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ==========================================
# # 1. LOAD THE AI MODEL ON STARTUP
# # ==========================================
# print("Loading TFLite model...")
# try:
#     # Make sure 'cry_detection_model.tflite' is in the same folder as main.py
#     interpreter = tf.lite.Interpreter(model_path="baby_cry_v2_pro.tflite")
#     interpreter.allocate_tensors()

#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()
#     print("✅ Model loaded successfully!")

# except Exception as e:
#     print(f"❌ Failed to load model. Is 'cry_detection_model.tflite' in the right folder? Error: {e}")


# # ==========================================
# # 2. DEFINE DATA STRUCTURES
# # ==========================================
# class SoundData(BaseModel):
#     intensity: int
#     status: str
#     duration: int = 0


# @app.get("/")
# def health_check():
#     return {"status": "API is live and ready", "model_loaded": "interpreter" in globals()}


# # ==========================================
# # 3. ENDPOINT: IoT SENSORS (ESP32/Arduino)
# # ==========================================
# @app.post("/sound-detected")
# async def process_sound_data(data: SoundData):
#     try:
#         is_crying = data.intensity > 70
#         print(f"Received IoT Data - Intensity: {data.intensity}, Status: {data.status}")

#         return {
#             "success": True,
#             "received_intensity": data.intensity,
#             "action": "Triggering React alerts" if is_crying else "Normal"
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# # ==========================================
# # 4. ENDPOINT: RAW AUDIO FOR AI PREDICTION
# # ==========================================
# # @app.post("/predict-media")
# # async def predict_media(file: UploadFile = File(...)):
# #     temp_audio_path = ""
# #     try:
# #         # 1. Save temporarily to disk to prevent librosa file-reading errors
# #         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
# #             temp_audio.write(await file.read())
# #             temp_audio_path = temp_audio.name

# #         # 2. Load audio at 22,050 Hz (Standard ML training sample rate)
# #         y, sr = librosa.load(temp_audio_path, sr=22050)

# #         # 3. Generate the Mel Spectrogram
# #         melspec = librosa.feature.melspectrogram(
# #             y=y,
# #             sr=sr,
# #             n_mels=40,         # 40 frequency bins
# #             n_fft=2048,
# #             hop_length=512     # Creates the 216 time frames over 5 seconds
# #         )

# #         # 4. Convert power to decibels (log scale)
# #         log_melspec = librosa.power_to_db(melspec, ref=np.max)

# #         # 5. Normalize between 0 and 1 (Standard scaling for Neural Networks)
# #         log_melspec = (log_melspec - np.min(log_melspec)) / (np.max(log_melspec) - np.min(log_melspec) + 1e-10)

# #         # 6. Ensure exactly 216 time frames
# #         if log_melspec.shape[1] < 216:
# #             pad_width = 216 - log_melspec.shape[1]
# #             log_melspec = np.pad(log_melspec, pad_width=((0, 0), (0, pad_width)), mode='constant')
# #         else:
# #             log_melspec = log_melspec[:, :216]

# #         # 7. Reshape to the exact tensor shape: [1, 40, 216, 1]
# #         real_input = log_melspec.reshape(1, 40, 216, 1).astype(np.float32)

# #         # 8. Run the TFLite Model
# #         interpreter.set_tensor(input_details[0]['index'], real_input)
# #         interpreter.invoke()

# #         # 9. Get the result
# #         prediction = interpreter.get_tensor(output_details[0]['index'])
# #         cry_probability = float(prediction[0][0])
# #         print(cry_probability)
# #         return {
# #             "success": True,
# #             "filename": file.filename,
# #             "ai_analysis": {
# #                 "cry_probability": cry_probability,
# #                 "is_crying": cry_probability > 0.80
# #             }
# #         }
# #     except Exception as e:
# #         # print(f"Error during prediction: {e}")
# #         # raise HTTPException(status_code=500, detail=str(e))
# #         print("---!!! THE BACKEND CRASHED HERE !!!---")
# #         traceback.print_exc() # This is the magic line that shows the error
# #         return {"success": False, "error": str(e)}
# #     finally:
# #         # 10. Clean up the temporary file so your hard drive doesn't fill up!
# #         if temp_audio_path and os.path.exists(temp_audio_path):
# #             os.remove(temp_audio_path)

# # 2 - format not changed

# # @app.post("/predict-media")
# # async def predict_media(file: UploadFile = File(...)):
# #     temp_audio_path = ""
# #     try:
# #         # 1. Save the file
# #         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
# #             content = await file.read()
# #             temp_audio.write(content)
# #             temp_audio_path = temp_audio.name

# #         # 2. LOAD AUDIO MANUALLY (This avoids the NoBackendError)
# #         try:
# #             # Read using soundfile instead of librosa.load
# #             data, samplerate = sf.read(temp_audio_path)
            
# #             # If stereo (2 channels), convert to mono
# #             if len(data.shape) > 1:
# #                 data = np.mean(data, axis=1)
            
# #             # Resample to exactly 22050Hz for the model
# #             y = librosa.resample(data, orig_sr=samplerate, target_sr=22050)
# #             sr = 22050
# #         except Exception as e:
# #             print(f"Manual load failed: {e}")
# #             raise e

# #         # 3. Create Mel Spectrogram (40 bins)
# #         melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, n_fft=2048, hop_length=512)
# #         log_melspec = librosa.power_to_db(melspec, ref=np.max)

# #         # 4. Normalize
# #         log_melspec = (log_melspec - np.min(log_melspec)) / (np.max(log_melspec) - np.min(log_melspec) + 1e-10)

# #         # 5. Fix Shape to exactly 216 frames
# #         if log_melspec.shape[1] < 216:
# #             pad_width = 216 - log_melspec.shape[1]
# #             log_melspec = np.pad(log_melspec, pad_width=((0, 0), (0, pad_width)), mode='constant')
# #         else:
# #             log_melspec = log_melspec[:, :216]

# #         # 6. Inference
# #         real_input = log_melspec.reshape(1, 40, 216, 1).astype(np.float32)
# #         interpreter.set_tensor(input_details[0]['index'], real_input)
# #         interpreter.invoke()
# #         prediction = interpreter.get_tensor(output_details[0]['index'])
        
# #         cry_probability = float(prediction[0][0])
# #         print(f"SUCCESS! Cry Probability: {cry_probability}")

# #         return {
# #             "success": True, 
# #             "ai_analysis": {"cry_probability": cry_probability, "is_crying": cry_probability > 0.85}
# #         }

# #     except Exception as e:
# #         print("--- ERROR LOG ---")
# #         traceback.print_exc()
# #         return {"success": False, "error": str(e)}
# #     finally:
# #         if temp_audio_path and os.path.exists(temp_audio_path):
# #             os.remove(temp_audio_path)


# from pydub import AudioSegment
# import io

# @app.post("/predict-media")
# async def predict_media(file: UploadFile = File(...)):
#     temp_audio_path = ""
#     try:
#         # 1. Read the raw bytes directly from the upload
#         audio_bytes = await file.read()
#         print(f"Received audio: {len(audio_bytes)} bytes")

#         # 2. FORCE decode using pydub (This is the only way to handle WebM)
#         # We wrap the bytes in BytesIO so pydub can read it like a file
#         try:
#             audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
#         except Exception as e:
#             print("Pydub failed. Check if FFmpeg is installed and in PATH!")
#             raise e

#         # 3. Standardize the audio for the AI
#         audio = audio.set_frame_rate(22050).set_channels(1)
        
#         # Convert to numpy array (float32)
#         samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        
#         # Normalization (Scale to -1.0 to 1.0)
#         y = samples / (np.max(np.abs(samples)) + 1e-10)
#         sr = 22050

#         # 4. Generate Mel Spectrogram (40 bins)
#         melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, n_fft=2048, hop_length=512)
#         log_melspec = librosa.power_to_db(melspec, ref=np.max)

#         # 5. Normalize (0 to 1)
#         log_melspec = (log_melspec - np.min(log_melspec)) / (np.max(log_melspec) - np.min(log_melspec) + 1e-10)

#         # 6. Ensure 216 time frames
#         if log_melspec.shape[1] < 216:
#             pad_width = 216 - log_melspec.shape[1]
#             log_melspec = np.pad(log_melspec, pad_width=((0, 0), (0, pad_width)), mode='constant')
#         else:
#             log_melspec = log_melspec[:, :216]

#         # 7. Final Tensor Shape: [1, 40, 216, 1]
#         real_input = log_melspec.reshape(1, 40, 216, 1).astype(np.float32)

#         # 8. Inference
#         interpreter.set_tensor(input_details[0]['index'], real_input)
#         interpreter.invoke()
#         prediction = interpreter.get_tensor(output_details[0]['index'])
        
#         cry_probability = float(prediction[0][0])
#         print(f">>> Prediction Result: {cry_probability}")

#         return {
#             "success": True, 
#             "ai_analysis": {"cry_probability": cry_probability, "is_crying": cry_probability > 0.85}
#         }

#     except Exception as e:
#         print("---!!! PREDICTION ERROR !!!---")
#         traceback.print_exc()
#         return {"success": False, "error": str(e)}

# if __name__ == "__main__":
#     # Run the server
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
import os
import io
import librosa
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import traceback
from pydub import AudioSegment

app = FastAPI(title="Baby Monitor AI Backend")

# Configure CORS so React (port 5173) can talk to FastAPI (port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 1. LOAD THE AI MODEL
# ==========================================
MODEL_PATH = "baby_cry_v2_pro.tflite"
interpreter = None
input_details = None
output_details = None

def load_model():
    global interpreter, input_details, output_details
    print(f"Loading TFLite model from {MODEL_PATH}...")
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file {MODEL_PATH} not found!")
            
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("✅ AI Model loaded and Ready!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")

load_model()

@app.get("/")
def health_check():
    return {"status": "API is live", "model_loaded": interpreter is not None}

# ==========================================
# 2. THE PREDICT ENDPOINT
# ==========================================
@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    if interpreter is None:
        return {"success": False, "error": "Model not loaded"}

    try:
        # 1. Read bytes and decode
        audio_bytes = await file.read()
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))

        # 2. Preprocess: 22050Hz, Mono, Normalize
        audio = audio.set_frame_rate(22050).set_channels(1)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        y = samples / (np.max(np.abs(samples)) + 1e-10)

        # 3. Mel Spectrogram (Match training: 40 mels, 216 frames)
        melspec = librosa.feature.melspectrogram(y=y, sr=22050, n_mels=40, n_fft=2048, hop_length=512)
        log_melspec = librosa.power_to_db(melspec, ref=np.max)
        
        # Scale 0 to 1
        log_melspec = (log_melspec - np.min(log_melspec)) / (np.max(log_melspec) - np.min(log_melspec) + 1e-10)

        # Pad/Crop to exactly 216 frames
        if log_melspec.shape[1] < 216:
            log_melspec = np.pad(log_melspec, ((0, 0), (0, 216 - log_melspec.shape[1])), mode='constant')
        else:
            log_melspec = log_melspec[:, :216]

        # 4. Inference [1, 40, 216, 1]
        input_data = log_melspec.reshape(1, 40, 216, 1).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        prob = float(prediction[0][0])
        print(f"Result: {prob:.4f}")
        return {"success": True, "probability": prob, "is_crying": prob > 0.85}

    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)