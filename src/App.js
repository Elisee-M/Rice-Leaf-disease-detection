import React, { useState, useRef, useEffect } from 'react';
import * as ort from 'onnxruntime-web';
import './App.css';

function App() {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [session, setSession] = useState(null);
  const [classNames, setClassNames] = useState([]);
  const fileInputRef = useRef(null);

  useEffect(() => {
    const loadModel = async () => {
      try {
        const response = await fetch('/model/class_names.json');
        const classes = await response.json();
        setClassNames(classes);
        
        const ortSession = await ort.InferenceSession.create('/model/rice_model.onnx');
        setSession(ortSession);
        console.log('✅ Model loaded');
      } catch(error) {
        console.error('Error:', error);
      }
    };
    loadModel();
  }, []);

  const preprocessImage = (imgElement) => {
    const canvas = document.createElement('canvas');
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imgElement, 0, 0, 224, 224);
    
    const imageData = ctx.getImageData(0, 0, 224, 224);
    const input = new Float32Array(1 * 3 * 224 * 224);
    
    for (let i = 0; i < 224 * 224; i++) {
      input[0 * 224 * 224 + i] = (imageData.data[i * 4] / 255 - 0.485) / 0.229;
      input[1 * 224 * 224 + i] = (imageData.data[i * 4 + 1] / 255 - 0.456) / 0.224;
      input[2 * 224 * 224 + i] = (imageData.data[i * 4 + 2] / 255 - 0.406) / 0.225;
    }
    return input;
  };

  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    if (!file || !session) return;
    
    setLoading(true);
    const img = new Image();
    img.onload = async () => {
      try {
        const inputData = preprocessImage(img);
        const tensor = new ort.Tensor('float32', inputData, [1, 3, 224, 224]);
        const results = await session.run({ input: tensor });
        const output = results.output.data;
        const predictedIdx = output.indexOf(Math.max(...output));
        
        setPrediction({
          disease: classNames[predictedIdx],
          confidence: output[predictedIdx]
        });
      } catch(error) {
        console.error('Prediction error:', error);
      }
      setLoading(false);
    };
    img.src = URL.createObjectURL(file);
  };

  return (
    <div className="container">
      <h1>🌾 Rice Leaf Disease Detector</h1>
      <p>Powered by DenseNet201 - ONNX Runtime</p>
      
      <div className="upload-area" onClick={() => fileInputRef.current?.click()}>
        <input type="file" ref={fileInputRef} onChange={handleImageUpload} accept="image/*" style={{ display: 'none' }} />
        <p>📤 Click to upload a rice leaf image</p>
      </div>
      
      {loading && <p className="loading">🔬 Analyzing...</p>}
      
      {prediction && (
        <div className="result">
          <h2>{prediction.disease}</h2>
          <p>Confidence: {(prediction.confidence * 100).toFixed(2)}%</p>
        </div>
      )}
    </div>
  );
}

export default App;