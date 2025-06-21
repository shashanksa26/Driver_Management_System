import cv2
import numpy as np
import tensorflow as tf
import os

# Use only the quantized model
model_path = 'models/mobilefacenet_quantized.tflite'
print(f"[INFO] Using quantized model: {model_path}")

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print embedding size for debug
embedding_shape = output_details[0]['shape']
print(f"[INFO] Model embedding output shape: {embedding_shape}")

# Always use float32 for embeddings

def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (160, 160))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = (face_img.astype(np.float32) - 127.5) * 0.0078125
    return np.expand_dims(face_img, axis=0)

def extract_embedding(face_img):
    input_data = preprocess_face(face_img)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    embedding = interpreter.get_tensor(output_details[0]['index'])[0]
    # Ensure float32 and flatten
    embedding = np.array(embedding, dtype=np.float32).flatten()
    return embedding 