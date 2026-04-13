from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Baby Monitor AI Backend - Test Mode")

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# HEALTH CHECK ENDPOINT
# ==========================================
@app.get("/")
def health_check():
    return {
        "status": "API is live and ready",
        "mode": "test-mode (TensorFlow installation issue - using mock predictions)",
        "port": 8000
    }


# ==========================================
# AUDIO PREDICTION ENDPOINT (TEST)
# ==========================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Test endpoint that returns mock predictions while we fix TensorFlow.
    The return format matches what the frontend expects.
    """
    try:
        print(f"📝 Received audio: {file.filename}")
        
        # Mock prediction (in real implementation, this uses the TFLite model)
        # Randomly return a crying or normal result for testing
        import random
        mock_probability = random.uniform(0.2, 0.95)
        
        return {
            "success": True,
            "filename": file.filename,
            "probability": mock_probability,
            "is_crying": mock_probability > 0.80,
            "label": "Crying" if mock_probability > 0.80 else "Normal",
            "confidence": mock_probability,
            "mode": "TEST - Returns Mock Data"
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            "success": False,
            "error": str(e),
            "label": "Error",
            "confidence": 0.0
        }


# ==========================================
# RUN THE SERVER
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Baby Monitor AI Backend - TEST MODE")
    print("="*60)
    print("⚠️  Running in test mode (mock predictions)")
    print("📍 Server running at: http://localhost:8000")
    print("📚 API docs at: http://localhost:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
