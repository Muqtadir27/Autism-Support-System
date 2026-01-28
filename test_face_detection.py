#!/usr/bin/env python
"""
Test face detection with different methods
"""
import cv2
import numpy as np

print("Testing face detection methods\n")

# Load cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
print(f"Haar Cascade loaded: {not face_cascade.empty()}")

# Open webcam
cap = cv2.VideoCapture(0)
print(f"Webcam opened: {cap.isOpened()}")

if not cap.isOpened():
    print("ERROR: Cannot open webcam")
    exit(1)

frame_count = 0
face_counts = []

print("\nCapturing 10 frames for face detection...\n")

while frame_count < 10:
    ret, frame = cap.read()
    if not ret:
        print("Cannot read frame")
        break
    
    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
    face_counts.append(len(faces))
    
    print(f"Frame {frame_count}: {len(faces)} faces detected")
    
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            print(f"  - Face at ({x},{y}) size {w}x{h}")

cap.release()

print(f"\nTotal frames: {len(face_counts)}")
print(f"Average faces per frame: {np.mean(face_counts):.1f}")
print(f"Frames with detected faces: {sum(1 for c in face_counts if c > 0)}/10")
