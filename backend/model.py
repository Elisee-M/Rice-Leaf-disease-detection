import tensorflow as tf
import numpy as np
import json
import os

class RiceDiseaseDetector:
    def __init__(self, model_path=None, class_names_path=None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'rice_disease_model.h5')
        if class_names_path is None:
            class_names_path = os.path.join(os.path.dirname(__file__), 'class_names.json')

        self.model = tf.keras.models.load_model(model_path)
        with open(class_names_path, 'r') as f:
            self.class_names = json.load(f)
        self.img_size = 224

    def predict(self, image):
        resized = tf.image.resize(image, (self.img_size, self.img_size))
        normalized = resized / 255.0
        input_batch = np.expand_dims(normalized, axis=0)

        predictions = self.model.predict(input_batch, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))

        return {
            'disease': self.class_names[predicted_class],
            'confidence': confidence,
            'all_scores': {self.class_names[i]: float(predictions[0][i]) for i in range(len(self.class_names))}
        }
