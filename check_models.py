#!/usr/bin/env python
"""
Quick test to check if the models load and what their properties are
"""
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

print("="*80)
print("CHECKING MODEL FILES AND PROPERTIES")
print("="*80)

# Check Caffe model files
current_dir = os.path.dirname(os.path.abspath(__file__))
prototxt_path = os.path.join(current_dir, "predict", "Autismfiles", "deploy.prototxt.txt")
caffemodel_path = os.path.join(current_dir, "predict", "Autismfiles", "res10_300x300_ssd_iter_140000.caffemodel")
emotion_model_path = os.path.join(current_dir, "predict", "Autismfiles", "fer2013_mini_XCEPTION.102-0.66.hdf5")

print("\nFile Existence Checks:")
print(f"  Prototxt: {os.path.exists(prototxt_path)} ({prototxt_path})")
print(f"  CaffeModel: {os.path.exists(caffemodel_path)} ({caffemodel_path})")
print(f"  Emotion Model: {os.path.exists(emotion_model_path)} ({emotion_model_path})")

if os.path.exists(caffemodel_path):
    print(f"  CaffeModel size: {os.path.getsize(caffemodel_path) / 1024 / 1024:.2f} MB")
if os.path.exists(emotion_model_path):
    print(f"  Emotion Model size: {os.path.getsize(emotion_model_path) / 1024 / 1024:.2f} MB")

print("\nLoading Face Detection Model (Caffe)...")
try:
    face_net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
    print("  ✓ Face detection model loaded successfully")
except Exception as e:
    print(f"  ✗ FAILED: {e}")

print("\nLoading Emotion Recognition Model (Keras)...")
try:
    emotion_net = load_model(emotion_model_path, compile=False)
    print("  ✓ Emotion model loaded successfully")
    print(f"  - Input shape: {emotion_net.input_shape}")
    print(f"  - Output shape: {emotion_net.output_shape}")
    print(f"  - Model summary:")
    emotion_net.summary()
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting Face Detection...")
try:
    # Create a dummy image
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    h, w = dummy_frame.shape[:2]
    blob = cv2.dnn.blobFromImage(dummy_frame, 1.0, (300, 300), [104, 117, 123], False, False)
    face_net.setInput(blob)
    detections = face_net.forward()
    print(f"  ✓ Face detection works, detections shape: {detections.shape}")
except Exception as e:
    print(f"  ✗ FAILED: {e}")

print("\nTesting Emotion Prediction...")
try:
    # Create a dummy 64x64 grayscale image
    dummy_face = np.random.random((1, 64, 64, 1)).astype('float32')
    prediction = emotion_net.predict(dummy_face, verbose=0)
    print(f"  ✓ Emotion prediction works, output shape: {prediction.shape}")
    print(f"  - Output values: {prediction[0]}")
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    top_emotion = emotion_labels[np.argmax(prediction[0])]
    print(f"  - Predicted emotion: {top_emotion}")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
