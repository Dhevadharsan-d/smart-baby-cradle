



import { useState, useRef } from 'react';
import { Mic, Activity, Loader2 } from 'lucide-react';

export default function LocalMonitor() {
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [result, setResult] = useState<{ label: string; confidence: number } | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream);
    mediaRecorderRef.current = recorder;
    const chunks: Blob[] = [];

    recorder.ondataavailable = (e) => chunks.push(e.data);
    recorder.onstop = async () => {
      const blob = new Blob(chunks, { type: 'audio/wav' });
      await sendToAI(blob);
    };

    recorder.start();
    setIsMonitoring(true);

    // Record for 5 seconds then stop automatically
    setTimeout(() => {
      recorder.stop();
      stream.getTracks().forEach(t => t.stop());
      setIsMonitoring(false);
    }, 5000);
  };

  // const sendToAI = async (blob: Blob) => {
  //   setIsProcessing(true);
  //   const formData = new FormData();
  //   formData.append('file', blob, 'clip.wav');

  //   try {
  //     const res = await fetch('http://localhost:8000/predict', {
  //       method: 'POST',
  //       body: formData,
  //     });
  //     const data = await res.json();
  //     setResult({ label: data.label, confidence: data.confidence });
  //   } catch (err) {
  //     console.error("AI Server Error:", err);
  //   } finally {
  //     setIsProcessing(false);
  //   }
  // };

  const sendToAI = async (audioBlob: Blob) => {
  try {
    const formData = new FormData();
    // 'file' must match the backend's variable name
    formData.append('file', audioBlob, 'recording.webm');

    const response = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      body: formData,
    });

    const result = await response.json();
    
    if (result.success) {
      console.log("AI Prediction Probability:", result.probability);
      // Here you would update your React state: 
      // setStatus(result.is_crying ? "Crying" : "Normal");
    } else {
      console.error("Backend error:", result.error);
    }
  } catch (error) {
    console.error("Failed to connect to AI Backend. Is main.py running?", error);
  }
};

  return (
    <div className="p-6 bg-white rounded-2xl shadow-xl border border-indigo-100">
      <div className="flex justify-between items-center mb-6">
        <h3 className="font-bold text-gray-800">Local AI Tester</h3>
        {isMonitoring && <Activity className="text-red-500 animate-pulse" />}
      </div>

      <button
        onClick={startRecording}
        disabled={isMonitoring || isProcessing}
        className="w-full bg-indigo-600 text-white py-4 rounded-xl font-bold flex items-center justify-center space-x-2 disabled:opacity-50"
      >
        {isProcessing ? <Loader2 className="animate-spin" /> : <Mic size={20} />}
        <span>{isMonitoring ? 'Recording 5s...' : 'Test 5s Sample'}</span>
      </button>

      {result && (
        <div className={`mt-6 p-4 rounded-xl border-2 ${result.label === 'Crying' ? 'bg-red-50 border-red-200' : 'bg-green-50 border-green-200'}`}>
          <p className="text-sm font-bold text-gray-500 uppercase tracking-wider">Result</p>
          <p className={`text-2xl font-black ${result.label === 'Crying' ? 'text-red-600' : 'text-green-600'}`}>
            {result.label} ({result.confidence}%)
          </p>
        </div>
      )}
    </div>
  );
}