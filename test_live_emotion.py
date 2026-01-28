#!/usr/bin/env python
"""
Alternative emotion detection using DeepFace as backup
"""
import os
import sys
import cv2

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mini.settings')
import django
django.setup()

print("Testing emotion detection with live webcam...")
print("Press SPACE to capture and analyze, Q to quit")

try:
    from deepface import DeepFace
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        sys.exit(1)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow('Press SPACE to capture, Q to quit', frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            # Analyze emotion
            try:
                print("\nAnalyzing frame...")
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
                if result and len(result) > 0:
                    emotion = result[0]['dominant_emotion']
                    print(f"Detected Emotion: {emotion}")
                    print(f"All emotions: {result[0]['emotion']}")
            except Exception as e:
                print(f"Error: {e}")
    
    cap.release()
    cv2.destroyAllWindows()
    
except ImportError:
    print("DeepFace not available, testing tensorflow model directly...")
    
    # Test the TensorFlow model
    import numpy as np
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array
    
    model_path = "predict/Autismfiles/fer2013_mini_XCEPTION.102-0.66.hdf5"
    emotion_net = load_model(model_path, compile=False)
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print("Using Cascade classifier for faces...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        cv2.imshow('Detect emotions - Press Q to quit', frame)
        
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (64, 64))
                face = face.astype('float32') / 255.0
                face = img_to_array(face)
                face = np.expand_dims(face, axis=0)
                
                pred = emotion_net.predict(face, verbose=0)
                emotion_idx = np.argmax(pred[0])
                emotion = emotion_labels[emotion_idx]
                conf = pred[0][emotion_idx]
                
                print(f"Emotion: {emotion} ({conf:.2f})")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

print("Test completed")
