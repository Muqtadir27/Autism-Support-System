import cv2
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

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)

# Twilio configuration
account_sid = 'AC56dc47b7bb55ca14c447b9954c6ab34c'
auth_token = 'f2ff2918faec0cf461acaf0752319fdf'
twilio_phone_number = '+12295866437'
recipient_phone_number = '+918897930902'

client = Client(account_sid, auth_token)

def initialize_models():
    # Get the absolute paths of the model files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prototxt_path = os.path.join(current_dir, "Autismfiles/deploy.prototxt.txt")
    caffemodel_path = os.path.join(current_dir, "Autismfiles/res10_300x300_ssd_iter_140000.caffemodel")
    emotion_model_path = os.path.join(current_dir, "Autismfiles/fer2013_mini_XCEPTION.102-0.66.hdf5")

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
    message = client.messages.create(
        body=f"Frequent distress detected: {emotion}",
        from_=twilio_phone_number,
        to=recipient_phone_number
    )
    print(f"Sent SMS notification: {message.sid}")

def log_emotion(emotion_log, emotion):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    emotion_log.append({"timestamp": timestamp, "emotion": emotion})

def save_emotion_log(emotion_log):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(current_dir, "emotion_log.xlsx")
    df = pd.DataFrame(emotion_log)
    df.to_excel(log_file, index=False)
    print(f"Emotion log saved to {log_file}")

def tts_worker(queue, emotion_actions):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)

    while True:
        emotion = queue.get()
        if emotion is None:
            break
        action = emotion_actions.get(emotion, "No specific action suggested.")
        engine.say(action)
        engine.runAndWait()
        queue.task_done()

def Autism_emotion_recognition():
    net, emotion_net = initialize_models()

    # Initialize the video stream
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Initialize text-to-speech thread
    tts_queue = Queue()
    threading.Thread(target=tts_worker, args=(tts_queue, emotion_actions), daemon=True).start()

    # Initialize emotion log
    emotion_log = []
    distress_count = 0
    emotion_buffer = deque(maxlen=15)  # Buffer to store last 15 detected emotions

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break  # Exit the loop if frame capture fails

        # Get the frame dimensions
        (h, w) = frame.shape[:2]

        # Preprocess the frame: resize and create a blob
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        # Pass the blob through the network and obtain the detections
        net.setInput(blob)
        detections = net.forward()

        # Loop over the detections
        for i in range(0, detections.shape[2]):
            # Extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections by ensuring the confidence is greater than a threshold
            if confidence > 0.5:
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
                face_roi = cv2.resize(face_roi, (64, 64))
                face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                face_roi = face_roi.astype("float") / 255.0
                face_roi = np.expand_dims(face_roi, axis=-1)
                face_roi = np.expand_dims(face_roi, axis=0)

                # Predict emotion using HDF5 model
                emotion_preds = emotion_net.predict(face_roi, verbose=0)
                emotion_idx = np.argmax(emotion_preds)
                emotion = emotion_labels[emotion_idx]

                # Add the detected emotion to the buffer
                emotion_buffer.append(emotion)

                # Determine the most frequent emotion in the buffer
                most_common_emotion = Counter(emotion_buffer).most_common(1)[0][0]

                # Log the detected emotion
                log_emotion(emotion_log, most_common_emotion)

                # Draw the bounding box around the face along with the emotion label
                text = f"{most_common_emotion}"
                y = startY - 10 if startY - 10 > 10 else startY + 10
                color = emotion_colors.get(most_common_emotion, (0, 255, 0))
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.putText(frame, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                # Queue the TTS action
                tts_queue.put(most_common_emotion)

                # Handle distress notification
                if most_common_emotion in ['angry', 'fear', 'sad']:
                    distress_count += 1
                    if distress_count >= 6:  # Adjust threshold as needed
                        send_sms_notification(client, twilio_phone_number, recipient_phone_number, most_common_emotion)
                        distress_count = 0  # Reset distress count

        # Display the resulting frame
        cv2.imshow("Frame", frame)

        # Ensure the frame is displayed properly
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Stop the TTS worker
    tts_queue.put(None)

    # Save the emotion log to a file
    save_emotion_log(emotion_log)

    # Release the capture and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    return emotion_log
