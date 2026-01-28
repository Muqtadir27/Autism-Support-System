#!/usr/bin/env python
"""
Test script to verify emotion detection is working correctly
"""
import os
import sys
import cv2
import numpy as np
from io import BytesIO

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mini.settings')
import django
django.setup()

from predict.AutismEmoRec import process_single_frame_for_emotion

def test_with_webcam():
    """Test emotion detection with live webcam"""
    print("\n" + "="*80)
    print("LIVE WEBCAM EMOTION DETECTION TEST")
    print("="*80)
    print("\nStarting webcam... Press 'q' to quit and save frame for testing")
    print("Make different facial expressions to test emotion detection\n")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Cannot open webcam!")
        return
    
    frame_count = 0
    test_frame = None
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("ERROR: Cannot read from webcam!")
            break
        
        frame_count += 1
        cv2.imshow('Press Q to capture frame for emotion detection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            test_frame = frame
            print(f"\nFrame captured ({frame.shape})")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if test_frame is None:
        print("No frame captured. Exiting.")
        return
    
    # Save the frame to a file
    test_file_path = os.path.join(os.path.dirname(__file__), 'test_frame.jpg')
    cv2.imwrite(test_file_path, test_frame)
    print(f"Frame saved to: {test_file_path}")
    
    # Test emotion detection with the captured frame
    print("\n" + "="*80)
    print("TESTING EMOTION DETECTION WITH CAPTURED FRAME")
    print("="*80)
    
    # Convert frame to bytes (simulating file upload)
    success, buffer = cv2.imencode('.jpg', test_frame)
    if not success:
        print("ERROR: Could not encode frame to bytes")
        return
    
    frame_bytes = BytesIO(buffer.tobytes())
    frame_bytes.name = 'test_frame.jpg'
    
    # Call the emotion detection function
    emotion = process_single_frame_for_emotion(frame_bytes)
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULT: {emotion}")
    print(f"{'='*80}\n")

def test_with_sample_image(image_path):
    """Test emotion detection with a provided image file"""
    if not os.path.exists(image_path):
        print(f"ERROR: Image file not found: {image_path}")
        return
    
    print("\n" + "="*80)
    print(f"TESTING EMOTION DETECTION WITH: {image_path}")
    print("="*80 + "\n")
    
    # Read and prepare the image
    with open(image_path, 'rb') as f:
        image_file = BytesIO(f.read())
        image_file.name = os.path.basename(image_path)
        
        # Call the emotion detection function
        emotion = process_single_frame_for_emotion(image_file)
        
        print(f"\n{'='*80}")
        print(f"FINAL RESULT: {emotion}")
        print(f"{'='*80}\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Test with provided image
        test_with_sample_image(sys.argv[1])
    else:
        # Test with webcam
        test_with_webcam()
