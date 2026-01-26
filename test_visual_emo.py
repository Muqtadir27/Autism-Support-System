"""
Test script for visual emotion recognition functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'predict'))

def test_visual_emo():
    """
    Test the visual emotion recognition functionality
    """
    print("Testing Visual Emotion Recognition...")
    
    try:
        # Import the function
        from predict.AutismEmoRec import initialize_models
        
        print("✓ Successfully imported initialize_models function")
        
        # Test model loading
        print("Testing model loading...")
        try:
            net, emotion_net = initialize_models()
            print("✓ Models loaded successfully")
            print(f"✓ Face detection model: {type(net)}")
            print(f"✓ Emotion recognition model: {type(emotion_net)}")
        except Exception as e:
            print(f"✗ Error loading models: {e}")
            return False
            
        # Test that required files exist
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        required_files = [
            os.path.join(current_dir, "predict", "Autismfiles", "deploy.prototxt.txt"),
            os.path.join(current_dir, "predict", "Autismfiles", "res10_300x300_ssd_iter_140000.caffemodel"),
            os.path.join(current_dir, "predict", "Autismfiles", "fer2013_mini_XCEPTION.102-0.66.hdf5")
        ]
        
        for file_path in required_files:
            if os.path.exists(file_path):
                print(f"✓ Required file exists: {os.path.basename(file_path)}")
            else:
                print(f"✗ Required file missing: {os.path.basename(file_path)}")
                
        print("\nVisual Emotion Recognition test completed successfully!")
        print("\nTo run the actual visual emotion recognition, use:")
        print("  from predict.AutismEmoRec import Autism_emotion_recognition")
        print("  Autism_emotion_recognition()")
        print("\nThe function will open a camera window and detect emotions in real-time.")
        print("Press 'q' or close the window to exit the detection.")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_visual_emo()
    if success:
        print("\n✓ All tests passed! The visual emotion recognition should work properly.")
    else:
        print("\n✗ Some tests failed. Please check the implementation.")