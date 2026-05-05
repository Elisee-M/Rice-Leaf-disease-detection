import React, { useState, useRef, useEffect } from 'react';
import * as ort from 'onnxruntime-web';
import './App.css';

function App() {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [session, setSession] = useState(null);
  const [classNames, setClassNames] = useState([]);
  const [modelReady, setModelReady] = useState(false);
  const fileInputRef = useRef(null);

  useEffect(() => {
    const loadModel = async () => {
      try {
        console.log('Loading class names...');
        const response = await fetch('/model/class_names.json');
        const classes = await response.json();
        setClassNames(classes);
        console.log('Classes loaded:', classes);
        
        console.log('Loading ONNX model...');
        const ortSession = await ort.InferenceSession.create('/model/rice_model.onnx');
        setSession(ortSession);
        setModelReady(true);
        console.log('✅ Model loaded successfully');
      } catch(error) {
        console.error('Error loading model:', error);
        setModelReady(false);
      }
    };
    loadModel();
  }, []);

  const preprocessImage = (imgElement) => {
    return new Promise((resolve) => {
      const canvas = document.createElement('canvas');
      canvas.width = 224;
      canvas.height = 224;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(imgElement, 0, 0, 224, 224);
      
      const imageData = ctx.getImageData(0, 0, 224, 224);
      const input = new Float32Array(1 * 3 * 224 * 224);
      
      for (let i = 0; i < 224 * 224; i++) {
        // Normalize using ImageNet stats
        input[0 * 224 * 224 + i] = (imageData.data[i * 4] / 255 - 0.485) / 0.229;
        input[1 * 224 * 224 + i] = (imageData.data[i * 4 + 1] / 255 - 0.456) / 0.224;
        input[2 * 224 * 224 + i] = (imageData.data[i * 4 + 2] / 255 - 0.406) / 0.225;
      }
      
      resolve(input);
    });
  };

  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) {
      console.log('No file selected');
      return;
    }
    
    if (!session) {
      console.log('Model not loaded yet');
      alert('Model is still loading. Please wait.');
      return;
    }
    
    setLoading(true);
    setPrediction(null);
    
    const img = new Image();
    const imageUrl = URL.createObjectURL(file);
    
    img.onload = async () => {
      try {
        console.log('Image loaded, preprocessing...');
        const inputData = await preprocessImage(img);
        console.log('Preprocessing complete, shape:', inputData.length);
        
        const tensor = new ort.Tensor('float32', inputData, [1, 3, 224, 224]);
        console.log('Tensor created, running inference...');
        
        const results = await session.run({ input: tensor });
        console.log('Inference complete:', results);
        
        const output = results.output.data;
        console.log('Output scores:', Array.from(output));
        
        const predictedIdx = output.indexOf(Math.max(...output));
        const confidence = output[predictedIdx];
        
        console.log(`Predicted: ${classNames[predictedIdx]} with ${(confidence * 100).toFixed(2)}% confidence`);
        
        setPrediction({
          disease: classNames[predictedIdx],
          confidence: confidence,
          allScores: Array.from(output).map((score, idx) => ({
            name: classNames[idx],
            score: score
          }))
        });
      } catch(error) {
        console.error('Prediction error:', error);
        alert('Prediction failed. Check console for details.');
      } finally {
        setLoading(false);
        URL.revokeObjectURL(imageUrl);
      }
    };
    
    img.onerror = () => {
      console.error('Failed to load image');
      setLoading(false);
      URL.revokeObjectURL(imageUrl);
      alert('Failed to load image. Please try another file.');
    };
    
    img.src = imageUrl;
  };

  return (
    <div className="container">
      <h1>🌾 Rice Leaf Disease Detector</h1>
      <p>Powered by DenseNet201 - ONNX Runtime</p>
      
      <div className="model-status">
        Status: {modelReady ? '✅ Model ready' : '⏳ Loading model...'}
      </div>
      
      <div className="upload-area" onClick={() => fileInputRef.current?.click()}>
        <input 
          type="file" 
          ref={fileInputRef} 
          onChange={handleImageUpload} 
          accept="image/jpeg,image/png,image/jpg" 
          style={{ display: 'none' }} 
        />
        <p>📤 Click to upload a rice leaf image</p>
        <p style={{ fontSize: '12px', color: '#666' }}>Supports JPG, PNG</p>
      </div>
      
      {loading && (
        <div className="loading">
          <p>🔬 Analyzing image...</p>
        </div>
      )}
      
      {prediction && (
        <div className="result">
          <h2>{prediction.disease}</h2>
          <p>Confidence: {(prediction.confidence * 100).toFixed(2)}%</p>
          
          <h3>All Scores:</h3>
          {prediction.allScores.map((item, idx) => (
            <div key={idx} className="score-item">
              <span>{item.name}</span>
              <span>{(item.score * 100).toFixed(1)}%</span>
              <div className="progress-bar">
                <div 
                  className="progress-fill" 
                  style={{ width: `${item.score * 100}%` }}
                ></div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;