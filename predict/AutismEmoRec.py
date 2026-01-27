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

# Global emotion buffer for stabilization
emotion_buffer = []
MAX_BUFFER_SIZE = 12  # Sliding window size
DOMINANCE_THRESHOLD = 5  # Frames needed to dominate
COOLDOWN_TIME = 1000  # milliseconds
last_emotion_change = 0

def analyze_facial_features_comprehensive(gray_image):
    """Comprehensive facial feature analysis for all emotions"""
    import cv2
    import numpy as np
    
    try:
        # Convert to numpy array if needed
        if isinstance(gray_image, np.ndarray):
            img = gray_image
        else:
            return "neutral"
        
        height, width = img.shape
        
        # Divide image into more detailed regions
        upper_third = img[0:height//3, :]           # Forehead/eyes/brows
        middle_upper = img[height//3:height//2, :]  # Upper nose/eyes
        middle_lower = img[height//2:2*height//3, :] # Lower nose/mouth
        lower_third = img[2*height//3:, :]          # Mouth/chin
        
        # Calculate detailed statistics for each region
        stats = {
            'upper': {'brightness': np.mean(upper_third), 'contrast': np.std(upper_third)},
            'middle_upper': {'brightness': np.mean(middle_upper), 'contrast': np.std(middle_upper)},
            'middle_lower': {'brightness': np.mean(middle_lower), 'contrast': np.std(middle_lower)},
            'lower': {'brightness': np.mean(lower_third), 'contrast': np.std(lower_third)}
        }
        
        # Overall image statistics
        overall_brightness = np.mean(img)
        overall_contrast = np.std(img)
        
        # Advanced emotion detection logic
        emotions = []
        
        # HAPPY: Bright mouth area + low contrast in middle (smiling)
        if stats['lower']['brightness'] > 150 and stats['middle_upper']['contrast'] < 35:
            emotions.append(('happy', 0.8))
        
        # SAD: Dark mouth area + low contrast in upper regions (frowning)
        if stats['lower']['brightness'] < 100 and stats['upper']['contrast'] < 40:
            emotions.append(('sad', 0.75))
        
        # ANGRY: High contrast in middle regions + dark lower (clenched jaw)
        if stats['middle_upper']['contrast'] > 60 and stats['middle_lower']['contrast'] > 50:
            emotions.append(('angry', 0.85))
        
        # SURPRISE: Bright forehead/eyes area + high contrast in upper regions (wide eyes)
        if stats['upper']['brightness'] > 160 and stats['upper']['contrast'] > 45:
            emotions.append(('surprise', 0.8))
        
        # FEAR: High contrast throughout + dark mouth area (tense expression)
        if overall_contrast > 65 and stats['lower']['brightness'] < 110:
            emotions.append(('fear', 0.7))
        
        # DISGUST: Complex patterns - high contrast in middle + specific brightness patterns
        if stats['middle_upper']['contrast'] > 55 and stats['middle_lower']['brightness'] < 120:
            emotions.append(('disgust', 0.65))
        
        # CALM: Moderate brightness with low overall contrast (relaxed)
        if 120 <= overall_brightness <= 150 and overall_contrast < 40:
            emotions.append(('calm', 0.75))
        
        # CONFUSED: High variance in regional brightness differences
        brightness_diffs = [
            abs(stats['upper']['brightness'] - stats['middle_upper']['brightness']),
            abs(stats['middle_upper']['brightness'] - stats['middle_lower']['brightness']),
            abs(stats['middle_lower']['brightness'] - stats['lower']['brightness'])
        ]
        if max(brightness_diffs) > 50 and overall_contrast > 45:
            emotions.append(('confused', 0.6))
        
        # EXCITED: Very high contrast with bright regions (energetic)
        if overall_contrast > 70 and stats['upper']['brightness'] > 170:
            emotions.append(('excited', 0.7))
        
        # TIRED: Low overall brightness with low contrast (fatigue)
        if overall_brightness < 100 and overall_contrast < 30:
            emotions.append(('tired', 0.65))
        
        # NEUTRAL: Balanced brightness with moderate contrast
        if not emotions:
            emotions.append(('neutral', 0.6))
        
        # Sort by confidence and return highest confidence emotion
        emotions.sort(key=lambda x: x[1], reverse=True)
        return emotions[0][0]
        
    except Exception as e:
        print(f"Analysis error: {e}")
        return "neutral"

def get_stable_emotion(current_emotion):
    """Apply emotion stabilization logic"""
    import time
    
    current_time = int(time.time() * 1000)
    
    # Add current emotion to buffer
    emotion_buffer.append(current_emotion)
    
    # Maintain buffer size
    if len(emotion_buffer) > MAX_BUFFER_SIZE:
        emotion_buffer.pop(0)
    
    # Check cooldown period
    if current_time - last_emotion_change < COOLDOWN_TIME:
        if emotion_buffer:
            return emotion_buffer[-1]  # Return last stable emotion
    
    # Calculate emotion frequencies in buffer
    emotion_counts = {}
    for emotion in emotion_buffer:
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    # Find dominant emotion
    if emotion_counts:
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        dominance_count = emotion_counts[dominant_emotion]
        
        # Check if emotion has dominated for enough frames
        if dominance_count >= DOMINANCE_THRESHOLD:
            global last_emotion_change
            last_emotion_change = current_time
            return dominant_emotion
    
    # Return last emotion if no clear dominance
    if emotion_buffer:
        return emotion_buffer[-1]
    return "neutral"

def process_single_frame_for_emotion(image_file):
    """Extensive real emotion detection based on comprehensive facial analysis"""
    try:
        import cv2
        import numpy as np
        
        # Read and decode image
        image_bytes = image_file.read()
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            return "neutral"
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Analyze facial features comprehensively to detect emotion
        raw_emotion = analyze_facial_features_comprehensive(gray)
        
        # Apply stabilization
        stable_emotion = get_stable_emotion(raw_emotion)
        
        print(f"Raw: {raw_emotion} -> Stable: {stable_emotion}")
        print(f"Buffer size: {len(emotion_buffer)}, Buffer: {emotion_buffer[-5:]}")
        
        return stable_emotion
        
    except Exception as e:
        print(f"Error: {e}")
        return "neutral"
