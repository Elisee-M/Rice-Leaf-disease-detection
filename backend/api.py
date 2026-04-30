from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from model import RiceDiseaseDetector

app = FastAPI(title="Rice Disease Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = RiceDiseaseDetector()

@app.get("/")
def root():
    return {"message": "Rice Disease Detection API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = detector.predict(image)
    return result

@app.get("/classes")
def get_classes():
    return {"classes": detector.class_names}