#!/usr/bin/env python
"""
Test emotion detection with a simple test image
"""
import os
import sys
import cv2
import numpy as np
from io import BytesIO

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mini.settings')
import django
django.setup()

from predict.AutismEmoRec import process_single_frame_for_emotion, get_emotion_models

print("\n" + "="*80)
print("TESTING EMOTION DETECTION SYSTEM")
print("="*80)

# Test 1: Check if models load
print("\n[TEST 1] Loading models...")
try:
    models = get_emotion_models()
    print("✓ Models loaded successfully")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Test 2: Create a synthetic test image with a face
print("\n[TEST 2] Creating synthetic test image...")
try:
    # Create a test image (640x480, BGR)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Fill with a light background
    test_image[:] = [200, 200, 200]
    
    # Draw a simple face oval
    cv2.ellipse(test_image, (320, 240), (120, 150), 0, 0, 360, (150, 100, 80), -1)
    
    # Draw eyes
    cv2.circle(test_image, (280, 200), 15, (30, 30, 30), -1)
    cv2.circle(test_image, (360, 200), 15, (30, 30, 30), -1)
    
    # Save and encode
    success, buffer = cv2.imencode('.jpg', test_image)
    if not success:
        print("✗ Could not encode image")
        sys.exit(1)
    
    frame_bytes = BytesIO(buffer.tobytes())
    frame_bytes.name = 'test_face.jpg'
    
    print(f"✓ Test image created ({test_image.shape})")
    
    # Test emotion detection
    print("\n[TEST 3] Testing emotion detection...")
    emotion = process_single_frame_for_emotion(frame_bytes)
    print(f"\n✓ Detection result: {emotion}")
    
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("ALL TESTS PASSED")
print("="*80)
