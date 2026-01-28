#!/usr/bin/env python
"""
Direct test of emotion detection on a real webcam
"""
import os
import sys
import cv2
import numpy as np
from io import BytesIO

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mini.settings')
import django
django.setup()

from predict.AutismEmoRec import process_single_frame_for_emotion

print("Testing emotion detection with REAL WEBCAM")
print("Press SPACE to capture frame and test emotion, Q to quit\n")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open webcam")
    sys.exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot read frame")
        break
    
    cv2.imshow('EMOTION TEST - Press SPACE to capture, Q to quit', frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord(' '):
        # Save frame and test
        print(f"\n[CAPTURED] Frame size: {frame.shape}")
        
        # Encode to JPEG bytes (simulating what web camera sends)
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            print("Failed to encode frame")
            continue
        
        # Create file-like object
        frame_bytes = BytesIO(buffer.tobytes())
        frame_bytes.name = 'webcam_frame.jpg'
        
        # Test emotion detection
        print("[TESTING] Processing frame...")
        emotion = process_single_frame_for_emotion(frame_bytes)
        print(f"[RESULT] Emotion: {emotion}\n")

cap.release()
cv2.destroyAllWindows()
