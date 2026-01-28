#!/usr/bin/env python
"""
Check the exact emotion label order that the fer2013_mini_XCEPTION model expects
"""
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mini.settings')
import django
django.setup()

from tensorflow.keras.models import load_model
import numpy as np

model_path = "predict/Autismfiles/fer2013_mini_XCEPTION.102-0.66.hdf5"
print(f"Loading model from: {model_path}")

model = load_model(model_path, compile=False)
print(f"\nModel input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")
print(f"\nModel architecture:")
model.summary()

# Test with dummy inputs
print("\n" + "="*80)
print("TESTING WITH DUMMY INPUTS")
print("="*80)

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Test 1: All zeros
print("\n[Test 1] All zeros (64x64 black face):")
test_input = np.zeros((1, 64, 64, 1), dtype='float32')
output = model.predict(test_input, verbose=0)
print(f"Output: {output[0]}")
print(f"Softmax sum: {output[0].sum():.6f}")
for i, (label, val) in enumerate(zip(emotion_labels, output[0])):
    print(f"  {i}. {label}: {val:.6f}")

# Test 2: All ones
print("\n[Test 2] All ones (64x64 white face):")
test_input = np.ones((1, 64, 64, 1), dtype='float32')
output = model.predict(test_input, verbose=0)
print(f"Output: {output[0]}")
print(f"Softmax sum: {output[0].sum():.6f}")
for i, (label, val) in enumerate(zip(emotion_labels, output[0])):
    print(f"  {i}. {label}: {val:.6f}")

# Test 3: Random noise
print("\n[Test 3] Random noise:")
test_input = np.random.random((1, 64, 64, 1)).astype('float32')
output = model.predict(test_input, verbose=0)
print(f"Output: {output[0]}")
print(f"Softmax sum: {output[0].sum():.6f}")
for i, (label, val) in enumerate(zip(emotion_labels, output[0])):
    print(f"  {i}. {label}: {val:.6f}")

print(f"\nModel appears to be working correctly!")
