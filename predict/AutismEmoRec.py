# NOTE: OpenCV imports are deferred to avoid startup issues on servers without GUI support
# All cv2 imports happen inside functions only when needed

import numpy as np
import os
from tensorflow.keras.models import load_model # type: ignore
import threading
import pyttsx3
from datetime import datetime
from twilio.rest import Client
from queue import Queue
from collections import deque, Counter
import pandas as pd

# Suppress TensorFlow logging messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Define emotion labels and corresponding actions
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
emotion_actions = {
    'angry': "Try to calm down. Take deep breaths.",
    'disgust': "Focus on something pleasant.",
    'fear': "Remember, you are safe.",
    'happy': "Keep smiling and spread joy.",
    'sad': "It's okay to feel sad. Talk to someone you trust.",
    'surprise': "Enjoy the surprise and stay curious.",
    'neutral': "Stay calm and relaxed."
}

emotion_colors = {
    'angry': (0, 0, 255),       # Red
    'disgust': (0, 255, 0),     # Green
    'fear': (255, 0, 0),        # Blue
    'happy': (0, 255, 255),     # Yellow
    'sad': (255, 255, 0),       # Cyan
    'surprise': (255, 0, 255),  # Magenta
    'neutral': (255, 255, 255)  # White
}

# Initialize text-to-speech engine with error handling
try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
    TTS_AVAILABLE = True
except Exception as e:
    print(f"Text-to-speech not available: {e}")
    TTS_AVAILABLE = False

# Twilio configuration
account_sid = 'AC56dc47b7bb55ca14c447b9954c6ab34c'
auth_token = 'f2ff2918faec0cf461acaf0752319fdf'
twilio_phone_number = '+12295866437'
recipient_phone_number = '+918897930902'

client = Client(account_sid, auth_token)

def initialize_models():
    # Get the absolute paths of the model files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prototxt_path = os.path.join(current_dir, "Autismfiles", "deploy.prototxt.txt")
    caffemodel_path = os.path.join(current_dir, "Autismfiles", "res10_300x300_ssd_iter_140000.caffemodel")
    emotion_model_path = os.path.join(current_dir, "Autismfiles", "fer2013_mini_XCEPTION.102-0.66.hdf5")
    
    # Import cv2 and tensorflow.keras locally to avoid loading at startup
    import cv2
    from tensorflow.keras.models import load_model # type: ignore

    # Check if the files exist
    if not os.path.isfile(prototxt_path):
        raise FileNotFoundError(f"Prototxt file not found: {prototxt_path}")
    if not os.path.isfile(caffemodel_path):
        raise FileNotFoundError(f"Caffemodel file not found: {caffemodel_path}")
    if not os.path.isfile(emotion_model_path):
        raise FileNotFoundError(f"Emotion model file not found: {emotion_model_path}")

    # Load the face detection model
    net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

    # Load the emotion recognition model ignoring optimizer state
    emotion_net = load_model(emotion_model_path, compile=False)

    return net, emotion_net

def send_sms_notification(client, twilio_phone_number, recipient_phone_number, emotion):
    try:
        message = client.messages.create(
            body=f"Frequent distress detected: {emotion}",
            from_=twilio_phone_number,
            to=recipient_phone_number
        )
        print(f"Sent SMS notification: {message.sid}")
    except Exception as e:
        print(f"Failed to send SMS notification: {e}")

def log_emotion(emotion_log, emotion):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    emotion_log.append({"timestamp": timestamp, "emotion": emotion})

def log_single_emotion(emotion):
    """Log a single emotion detection to the emotion log file."""
    import os
    import pandas as pd
    from datetime import datetime
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(current_dir, "emotion_log.xlsx")
    
    # Create a new DataFrame with the single emotion
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = pd.DataFrame([{"timestamp": timestamp, "emotion": emotion}])
    
    # Append to existing log file or create new one
    if os.path.exists(log_file):
        try:
            existing_df = pd.read_excel(log_file)
            combined_df = pd.concat([existing_df, new_entry], ignore_index=True)
            combined_df.to_excel(log_file, index=False)
        except Exception as e:
            print(f"Error reading existing log, creating new one: {e}")
            new_entry.to_excel(log_file, index=False)
    else:
        new_entry.to_excel(log_file, index=False)
    
    print(f"Emotion '{emotion}' logged to {log_file}")

def save_emotion_log(emotion_log):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(current_dir, "emotion_log.xlsx")
    
    new_df = pd.DataFrame(emotion_log)
    
    if os.path.exists(log_file):
        try:
            existing_df = pd.read_excel(log_file)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_excel(log_file, index=False)
        except Exception as e:
            print(f"Error reading existing log, creating new one: {e}")
            new_df.to_excel(log_file, index=False)
    else:
        new_df.to_excel(log_file, index=False)
    
    print(f"Emotion log saved to {log_file}")

def tts_worker(queue, emotion_actions):
    # Initialize TTS engine with error handling
    try:
        import pyttsx3
        local_engine = pyttsx3.init()
        local_engine.setProperty('rate', 150)
        local_engine.setProperty('volume', 0.9)
        local_tts_available = True
    except Exception as e:
        print(f"Local TTS not available: {e}")
        local_tts_available = False
        
    while True:
        emotion = queue.get()
        if emotion is None:
            break
        action = emotion_actions.get(emotion, "No specific action suggested.")
        
        if local_tts_available:
            try:
                local_engine.say(action)
                local_engine.runAndWait()
            except Exception as e:
                print(f"TTS Error: {e}")
        else:
            print(f"TTS Unavailable - Would have said: {action}")
            
        queue.task_done()

def Autism_emotion_recognition():
    print("Initializing Autism Emotion Recognition System...")
    
    # Import cv2 locally to avoid loading at startup
    import cv2
    import platform
    import ctypes
    print("' OpenCV imported successfully")
    
    # Initialize models with error handling
    print("Loading AI models...")
    try:
        net, emotion_net = initialize_models()
        print("' AI models loaded successfully")
    except Exception as e:
        print(f"' Error loading models: {e}")
        return
        
    # Initialize the video stream with multiple camera indices
    print("Initializing camera...")
    cap = None
    camera_index = 0
    
    # Try different camera indices
    for i in range(5):  # Try more camera indices
        print(f"Trying camera index {i}...")
        try:
            temp_cap = cv2.VideoCapture(i)
            # Set camera properties for better performance
            temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            temp_cap.set(cv2.CAP_PROP_FPS, 30)
            
            if temp_cap.isOpened():
                ret, test_frame = temp_cap.read()
                if ret and test_frame is not None:
                    print(f"' Camera {i} working properly")
                    cap = temp_cap
                    camera_index = i
                    break
                else:
                    temp_cap.release()
            else:
                print(f"Camera index {i} not available")
        except Exception as e:
            print(f"Error trying camera index {i}: {e}")
            continue
    
    if cap is None:
        print("' Error: Could not open any camera.")
        print("Please check:")
        print("1. Camera is connected and not in use by another application")
        print("2. Camera permissions are granted")
        print("3. Try connecting an external webcam")
        return
        
    print(f"' Using camera index {camera_index}")

    # Initialize text-to-speech thread
    print("Initializing Text-to-Speech...")
    tts_queue = Queue()
    tts_thread = threading.Thread(target=tts_worker, args=(tts_queue, emotion_actions), daemon=True)
    tts_thread.start()
    print("' TTS system initialized")

    # Initialize emotion log
    emotion_log = []
    distress_count = 0
    emotion_buffer = deque(maxlen=15)  # Buffer to store last 15 detected emotions
    
    print("' Emotion recognition system ready!")
    print("Press 'q' or close the window to exit")
    print("-" * 50)
    
    # Create OpenCV window explicitly before the loop to ensure it appears
    window_name = "Autism Emotion Recognition - Press 'q' to quit"
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        # Create a black frame to show immediately so window appears
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(black_frame, "Initializing camera...", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow(window_name, black_frame)
        cv2.waitKey(1)  # Force window to update
        
    except Exception as e:
        print(f"Warning: Could not create window: {e}")
        import traceback
        traceback.print_exc()
    
    frame_count = 0
    faces_detected = 0
    
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("' Failed to capture frame")
                break  # Exit the loop if frame capture fails

            frame_count += 1
            
            # Get the frame dimensions
            (h, w) = frame.shape[:2]

            # Preprocess the frame: resize and create a blob
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))

            # Pass the blob through the network and obtain the detections
            net.setInput(blob)
            detections = net.forward()

            current_faces = 0
            
            # Loop over the detections
            for i in range(0, detections.shape[2]):
                # Extract the confidence (i.e., probability) associated with the prediction
                confidence = detections[0, 0, i, 2]

                # Filter out weak detections by ensuring the confidence is greater than a threshold
                if confidence > 0.5:
                    current_faces += 1
                    
                    # Compute the (x, y)-coordinates of the bounding box for the face
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Ensure the bounding box is within the frame dimensions
                    startX, startY = max(0, startX), max(0, startY)
                    endX, endY = min(w, endX), min(h, endY)

                    # Extract the face ROI (Region of Interest)
                    face_roi = frame[startY:endY, startX:endX]

                    # Ensure the ROI is valid
                    if face_roi.size == 0:
                        continue

                    # Preprocess the face ROI for emotion recognition
                    face_roi_resized = cv2.resize(face_roi, (64, 64))
                    face_roi_gray = cv2.cvtColor(face_roi_resized, cv2.COLOR_BGR2GRAY)
                    face_roi_normalized = face_roi_gray.astype("float") / 255.0
                    face_roi_expanded = np.expand_dims(face_roi_normalized, axis=-1)
                    face_roi_batch = np.expand_dims(face_roi_expanded, axis=0)

                    try:
                        # Predict emotion using HDF5 model
                        emotion_preds = emotion_net.predict(face_roi_batch, verbose=0)
                        emotion_idx = np.argmax(emotion_preds)
                        emotion = emotion_labels[emotion_idx]

                        # Add the detected emotion to the buffer
                        emotion_buffer.append(emotion)

                        # Determine the most frequent emotion in the buffer
                        if len(emotion_buffer) > 0:
                            most_common_emotion = Counter(emotion_buffer).most_common(1)[0][0]
                        else:
                            most_common_emotion = emotion  # Fallback to current emotion

                        # Log the detected emotion
                        log_emotion(emotion_log, most_common_emotion)

                        # Draw the bounding box around the face along with the emotion label
                        text = f"{most_common_emotion}"
                        y = startY - 10 if startY - 10 > 10 else startY + 10
                        color = emotion_colors.get(most_common_emotion, (0, 255, 0))
                        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                        cv2.putText(frame, text, (startX, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                        # Queue the TTS action (only if TTS is available and not overwhelmed)
                        if TTS_AVAILABLE:
                            tts_queue.put(most_common_emotion)

                        # Handle distress notification
                        if most_common_emotion in ['angry', 'fear', 'sad']:
                            distress_count += 1
                            if distress_count >= 10:  # Increased threshold to reduce spam
                                send_sms_notification(client, twilio_phone_number, recipient_phone_number, most_common_emotion)
                                distress_count = 0  # Reset distress count
                                
                    except Exception as e:
                        print(f"Error in emotion prediction: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

            # Update face detection counter
            if current_faces > 0:
                faces_detected += 1
                
            # Display statistics every 30 frames
            if frame_count % 30 == 0:
                print(f"Frame: {frame_count}, Faces detected: {current_faces}, Total faces detected: {faces_detected}")

            # Display the resulting frame - Try to show the window
            try:
                cv2.imshow(window_name, frame)
            except cv2.error as e:
                if "The function is not implemented" in str(e):
                    print("GUI not available. Running in headless mode...")
                    print("Frame processing continuing without display...")
                    # Just continue processing without GUI
                    import time
                    time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                    continue  # Skip the key press check if GUI is not available
                else:
                    raise e  # Re-raise if it's a different error

            # Exit on 'q' key press or window close
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exiting emotion recognition...")
                break
            
            # Check if window was closed
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("Window closed by user...")
                    break
            except:
                # Window might have been destroyed
                break
                
    except Exception as e:
        print(f"' Error during emotion recognition: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup section
        print("Cleaning up resources...")
        
        try:
            # Stop the TTS worker
            if 'tts_queue' in locals():
                tts_queue.put(None)
                print("' TTS worker stopped")
        except Exception as e:
            print(f"Error stopping TTS worker: {e}")
            
        try:
            # Save the emotion log to a file
            if 'emotion_log' in locals() and emotion_log:
                save_emotion_log(emotion_log)
                print("' Emotion log saved")
        except Exception as e:
            print(f"Error saving emotion log: {e}")
            
        try:
            # Release the capture
            if 'cap' in locals() and cap is not None:
                cap.release()
                print("' Camera released")
        except Exception as e:
            print(f"Error releasing camera: {e}")
            
        try:
            # Close any OpenCV windows
            cv2.destroyAllWindows()
            print("' OpenCV windows closed")
        except cv2.error as e:
            if "The function is not implemented" in str(e):
                print("' GUI cleanup skipped (headless mode)")
            else:
                print(f"Error closing windows: {e}")
        except Exception as e:
            print(f"Error closing windows: {e}")
            
        print("' Cleanup completed")
        if 'emotion_log' in locals():
            return emotion_log
        else:
            return []

def detect_emotion_from_face_roi(face_roi, emotion_net):
    """
    Helper function to detect emotion from a face region of interest.
    """
    try:
        import cv2
        import numpy as np
        
        # Preprocess the face ROI for emotion recognition
        face_roi_resized = cv2.resize(face_roi, (64, 64))
        face_roi_gray = cv2.cvtColor(face_roi_resized, cv2.COLOR_BGR2GRAY)
        face_roi_normalized = face_roi_gray.astype("float") / 255.0
        face_roi_expanded = np.expand_dims(face_roi_normalized, axis=-1)
        face_roi_batch = np.expand_dims(face_roi_expanded, axis=0)

        # Predict emotion using HDF5 model
        emotion_preds = emotion_net.predict(face_roi_batch, verbose=0)
        emotion_idx = np.argmax(emotion_preds)
        emotion = emotion_labels[emotion_idx]
        
        # Add some confidence-based logic
        confidence = np.max(emotion_preds)
        print(f"Emotion prediction confidence: {confidence}")
        
        # If confidence is very low, return neutral
        if confidence < 0.3:
            print("Low confidence, returning neutral")
            return "neutral"
            
        return emotion
    except Exception as e:
        print(f"Error in emotion prediction: {e}")
        import traceback
        traceback.print_exc()
        return "neutral"

# Global variables for ML model
emotion_model = None
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def load_emotion_model():
    """Load pre-trained emotion recognition model"""
    global emotion_model
    try:
        import cv2
        import numpy as np
        from tensorflow.keras.models import load_model
        import os
        
        # Path to emotion model (you'll need to download this)
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'emotion_model.h5')
        
        if os.path.exists(model_path):
            emotion_model = load_model(model_path)
            print("Emotion model loaded successfully")
            return True
        else:
            print(f"Model file not found at: {model_path}")
            return False
    except Exception as e:
        print(f"Error loading emotion model: {e}")
        return False

def detect_emotion_ml(frame):
    """Real ML-based emotion detection"""
    try:
        import cv2
        import numpy as np
        
        if emotion_model is None:
            # Try to load model if not already loaded
            if not load_emotion_model():
                return "neutral"
        
        # Preprocess frame for emotion detection
        # Resize to model input size (typically 48x48 for emotion models)
        resized = cv2.resize(frame, (48, 48))
        
        # Convert to grayscale if needed
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized
            
        # Normalize pixel values
        normalized = gray.astype('float32') / 255.0
        
        # Reshape for model input (1, 48, 48, 1)
        input_data = normalized.reshape(1, 48, 48, 1)
        
        # Make prediction
        predictions = emotion_model.predict(input_data, verbose=0)
        
        # Get predicted emotion
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        emotion = emotion_labels[predicted_class]
        
        print(f"ML Prediction: {emotion} (confidence: {confidence:.2f})")
        return emotion
        
    except Exception as e:
        print(f"ML detection error: {e}")
        # Fallback to feature-based detection
        return detect_emotion_features(frame)

def detect_emotion_features(frame):
    """Feature-based emotion detection as backup"""
    import cv2
    import numpy as np
    
    try:
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        # Simple feature analysis
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Enhanced feature-based detection
        if brightness > 160 and contrast > 40:
            return "happy"
        elif brightness < 90 and contrast < 30:
            return "sad"
        elif contrast > 60:
            return "angry"
        elif brightness > 140:
            return "surprise"
        elif contrast > 40:
            return "fear"
        else:
            return "neutral"
            
    except Exception as e:
        print(f"Feature detection error: {e}")
        return "neutral"

# Simple CNN-based emotion detection
def simple_cnn_emotion_detection(frame):
    """Lightweight CNN-based emotion detection"""
    import cv2
    import numpy as np
    
    try:
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        # Resize to standard size
        resized = cv2.resize(gray, (48, 48))
        
        # Extract advanced features
        # Edge detection for facial feature analysis
        edges = cv2.Canny(resized, 50, 150)
        edge_density = np.sum(edges > 0) / (48 * 48)
        
        # Calculate regional statistics
        height, width = resized.shape
        upper_region = resized[0:height//3, :]
        middle_region = resized[height//3:2*height//3, :]
        lower_region = resized[2*height//3:, :]
        
        # Statistical features
        upper_mean = np.mean(upper_region)
        middle_mean = np.mean(middle_region)
        lower_mean = np.mean(lower_region)
        
        upper_std = np.std(upper_region)
        middle_std = np.std(middle_region)
        lower_std = np.std(lower_region)
        
        # Advanced feature vector
        features = [
            upper_mean/255.0,      # Normalized brightness features
            middle_mean/255.0,
            lower_mean/255.0,
            upper_std/255.0,       # Normalized contrast features
            middle_std/255.0,
            lower_std/255.0,
            edge_density,          # Edge density feature
            np.mean(resized)/255.0, # Overall brightness
            np.std(resized)/255.0   # Overall contrast
        ]
        
        # Simple neural network logic (simulated CNN)
        feature_vector = np.array(features)
        
        # Weighted decision logic (simulating trained weights)
        # Happy detection
        happy_score = 0
        if lower_mean > 150 and middle_std < 40:  # Bright mouth, low mid contrast
            happy_score += 0.8
        if edge_density < 0.1:  # Smooth facial features
            happy_score += 0.3
            
        # Sad detection
        sad_score = 0
        if lower_mean < 100 and upper_std < 35:  # Dark mouth, smooth upper
            sad_score += 0.7
        if middle_mean < 120:  # Overall darker
            sad_score += 0.4
            
        # Angry detection
        angry_score = 0
        if middle_std > 60 and lower_mean < 130:  # High mid contrast, dark lower
            angry_score += 0.8
        if edge_density > 0.15:  # Sharp features
            angry_score += 0.3
            
        # Surprise detection
        surprise_score = 0
        if upper_mean > 160 and upper_std > 45:  # Bright, high contrast upper
            surprise_score += 0.8
        if edge_density > 0.12:
            surprise_score += 0.2
            
        # Fear detection
        fear_score = 0
        if np.std(resized) > 70 and lower_mean < 110:  # High overall contrast, dark mouth
            fear_score += 0.7
            
        # Neutral detection
        neutral_score = 0.5  # Baseline
        
        # Create score dictionary
        scores = {
            'happy': happy_score,
            'sad': sad_score,
            'angry': angry_score,
            'surprise': surprise_score,
            'fear': fear_score,
            'neutral': neutral_score
        }
        
        # Add other emotions based on remaining patterns
        disgust_score = max(0, 0.6 - max(scores.values()))
        calm_score = 0.4 if 120 <= np.mean(resized) <= 150 else 0.2
        tired_score = 0.5 if np.mean(resized) < 100 and np.std(resized) < 30 else 0.1
        
        scores.update({
            'disgust': disgust_score,
            'calm': calm_score,
            'tired': tired_score
        })
        
        # Select emotion with highest score
        predicted_emotion = max(scores, key=scores.get)
        confidence = scores[predicted_emotion]
        
        print(f"CNN Prediction: {predicted_emotion} (score: {confidence:.2f})")
        return predicted_emotion
        
    except Exception as e:
        print(f"CNN detection error: {e}")
        return "neutral"

# Global emotion buffer for stabilization
emotion_buffer = []
MAX_BUFFER_SIZE = 6  # Very responsive
DOMINANCE_THRESHOLD = 2  # Quick changes allowed
COOLDOWN_TIME = 600  # Fast updates
last_emotion_change = 0

def get_stable_emotion(current_emotion):
    """Apply emotion stabilization logic"""
    import time
    global last_emotion_change
    
    current_time = int(time.time() * 1000)
    
    # Add current emotion to buffer
    emotion_buffer.append(current_emotion)
    
    # Maintain buffer size
    if len(emotion_buffer) > MAX_BUFFER_SIZE:
        emotion_buffer.pop(0)
    
    # Check cooldown period
    if current_time - last_emotion_change < COOLDOWN_TIME:
        if emotion_buffer:
            return emotion_buffer[-1]
    
    # Calculate emotion frequencies
    emotion_counts = {}
    for emotion in emotion_buffer:
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    # Find dominant emotion
    if emotion_counts:
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        dominance_count = emotion_counts[dominant_emotion]
        
        if dominance_count >= DOMINANCE_THRESHOLD:
            last_emotion_change = current_time
            return dominant_emotion
    
    if emotion_buffer:
        return emotion_buffer[-1]
    return "neutral"

# Global model cache - load once and reuse
_emotion_models_cache = None

def get_emotion_models():
    """Load and cache emotion detection models"""
    global _emotion_models_cache
    
    if _emotion_models_cache is not None:
        return _emotion_models_cache
    
    print("[INIT] Loading emotion detection models...")
    try:
        import cv2
        from tensorflow.keras.models import load_model  # type: ignore
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prototxt_path = os.path.join(current_dir, "Autismfiles", "deploy.prototxt.txt")
        caffemodel_path = os.path.join(current_dir, "Autismfiles", "res10_300x300_ssd_iter_140000.caffemodel")
        emotion_model_path = os.path.join(current_dir, "Autismfiles", "fer2013_mini_XCEPTION.102-0.66.hdf5")
        
        # Verify files exist
        for path, name in [(prototxt_path, "Prototxt"), (caffemodel_path, "CaffeModel"), (emotion_model_path, "Emotion Model")]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} not found at {path}")
        
        # Load face detector
        face_net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
        print("[INIT] [OK] Face detection model loaded")
        
        # Load emotion model
        emotion_net = load_model(emotion_model_path, compile=False)
        print(f"[INIT] [OK] Emotion model loaded - Input: {emotion_net.input_shape}, Output: {emotion_net.output_shape}")
        
        _emotion_models_cache = {
            'face_net': face_net,
            'emotion_net': emotion_net
        }
        
        return _emotion_models_cache
    
    except Exception as e:
        print(f"[INIT] [FAILED] FAILED TO LOAD MODELS: {e}")
        import traceback
        traceback.print_exc()
        raise

def process_single_frame_for_emotion(image_file):
    """Real emotion detection using DeepFace (most reliable) with fallback to local models"""
    print("\n" + "="*80)
    print("EMOTION DETECTION REQUEST")
    print("="*80)
    
    try:
        import cv2
        import numpy as np
        
        print(f"[1] Received image: {image_file.name}")
        
        # Read image bytes
        image_bytes = image_file.read()
        print(f"[2] Image size: {len(image_bytes)} bytes")
        
        # Decode image
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            print("[ERROR] Could not decode image")
            return "neutral"
        
        print(f"[3] Frame decoded: {frame.shape}")
        
        # TRY METHOD 1: DeepFace (most reliable)
        try:
            print("[4a] Attempting DeepFace analysis (most reliable)...")
            from deepface import DeepFace
            
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
            if result and len(result) > 0:
                dominant_emotion = result[0]['dominant_emotion']
                print(f"[4a] DeepFace detected: {dominant_emotion}")
                print(f"[4a] All emotions: {result[0]['emotion']}")
                log_single_emotion(dominant_emotion)
                print("="*80 + "\n")
                return dominant_emotion
            else:
                print("[4a] DeepFace returned no results")
        except ImportError:
            print("[4a] DeepFace not available, trying local models...")
        except Exception as e:
            print(f"[4a] DeepFace error: {e}")
        
        # METHOD 2: Local TensorFlow model with dual face detection
        print("[4b] Using local TensorFlow model...")
        from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore
        
        models = get_emotion_models()
        face_net = models['face_net']
        emotion_net = models['emotion_net']
        print("[4b] Models loaded from cache")
        
        h, w = frame.shape[:2]
        emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        best_emotion = None
        
        # METHOD 2A: Try Caffe face detector
        print("[5a] Detecting faces (Caffe CNN)...")
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
        face_net.setInput(blob)
        detections = face_net.forward()
        print(f"[5a] Caffe detection shape: {detections.shape}")
        
        # Process Caffe detections
        for i in range(min(detections.shape[2], 20)):
            confidence = detections[0, 0, i, 2]
            
            if confidence > 0.25:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 - x1 < 20 or y2 - y1 < 20:
                    continue
                
                face = frame[y1:y2, x1:x2]
                emotion = _predict_emotion_from_face(face, emotion_net, emotion_labels)
                print(f"[6a] Caffe face detected: {emotion}")
                
                if emotion and emotion != "neutral":
                    best_emotion = emotion
                    break
        
        if best_emotion:
            print(f"\n[RESULT] {best_emotion} (Caffe detection)")
            log_single_emotion(best_emotion)
            print("="*80 + "\n")
            return best_emotion
        
        # METHOD 2B: Haar Cascade fallback
        print("[5b] Detecting faces (Haar Cascade)...")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        
        print(f"[5b] Haar Cascade found {len(faces)} faces")
        
        if len(faces) > 0:
            for (x, y, w_face, h_face) in faces:
                face = frame[y:y+h_face, x:x+w_face]
                emotion = _predict_emotion_from_face(face, emotion_net, emotion_labels)
                print(f"[6b] Haar face detected: {emotion}")
                
                if emotion:
                    log_single_emotion(emotion)
                    print(f"\n[RESULT] {emotion} (Haar Cascade detection)")
                    print("="*80 + "\n")
                    return emotion
        
        print(f"\n[RESULT] neutral (no face detected)")
        print("="*80 + "\n")
        return "neutral"
        
    except Exception as e:
        print(f"\n[ERROR] Exception: {e}")
        import traceback
        traceback.print_exc()
        print("="*80 + "\n")
        return "neutral"

def _predict_emotion_from_face(face, emotion_net, emotion_labels):
    """Helper function to predict emotion from a face region"""
    try:
        import cv2
        import numpy as np
        from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore
        
        if face.size == 0:
            return None
        
        # Preprocess
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (64, 64))
        face_normalized = face_resized.astype('float32') / 255.0
        face_input = img_to_array(face_normalized)
        face_batch = np.expand_dims(face_input, axis=0)
        
        print(f"  [PRED] Input shape: {face_batch.shape}, min={face_normalized.min():.3f}, max={face_normalized.max():.3f}, mean={face_normalized.mean():.3f}")
        
        # Predict
        pred = emotion_net.predict(face_batch, verbose=0)
        print(f"  [PRED] Raw predictions: {pred[0]}")
        print(f"  [PRED] Prediction sum: {pred[0].sum():.4f}")
        print(f"  [PRED] Min: {pred[0].min():.4f}, Max: {pred[0].max():.4f}")
        
        emotion_idx = np.argmax(pred[0])
        emotion = emotion_labels[emotion_idx]
        emotion_score = pred[0][emotion_idx]
        
        print(f"  [PRED] Selected: {emotion} (score={emotion_score:.4f})")
        
        return emotion
    except Exception as e:
        print(f"  [PRED ERROR] {e}")
        import traceback
        traceback.print_exc()
        return None
