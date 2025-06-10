import cv2
import cv2.data
from deepface import DeepFace
import mediapipe as mp
import pygame
from collections import deque

def emotion_and_gesture_detection():
    # Local functions
    def play_sound(emotion):
        if emotion == 'angry':
            pygame.mixer.music.load(angry_song)
        elif emotion == 'disgust':
            pygame.mixer.music.load(disgust_song)
        elif emotion == 'fear':
            pygame.mixer.music.load(fear_song)
        elif emotion == 'happy':
            pygame.mixer.music.load(happy_song)
        elif emotion == 'sad':
            pygame.mixer.music.load(sad_song)
        elif emotion == 'surprise':
            pygame.mixer.music.load(surprise_song)
        elif emotion == 'neutral':
            pygame.mixer.music.load(neutral_song)
        pygame.mixer.music.play(-1)  # Loop indefinitely

    def stop_sound():
        pygame.mixer.music.stop()

    def show_quote(frame, quote, x, y):
        cv2.rectangle(frame, (x, y - 60), (x + 500, y - 20), (0, 0, 0), -1)  # Black background
        cv2.putText(frame, quote, (x + 10, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    def is_hand_closed(hand_landmarks):
        finger_tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP,
                       mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                       mp_hands.HandLandmark.RING_FINGER_TIP,
                       mp_hands.HandLandmark.PINKY_TIP]
        finger_dips = [mp_hands.HandLandmark.INDEX_FINGER_DIP,
                       mp_hands.HandLandmark.MIDDLE_FINGER_DIP,
                       mp_hands.HandLandmark.RING_FINGER_DIP,
                       mp_hands.HandLandmark.PINKY_DIP]
        closed = True
        for tip, dip in zip(finger_tips, finger_dips):
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[dip].y:
                closed = False
                break
        return closed

    def is_index_finger_pointing(hand_landmarks):
        if (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y <
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y and
            all(hand_landmarks.landmark[tip].y > hand_landmarks.landmark[dip].y
                for tip, dip in zip([mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                                     mp_hands.HandLandmark.RING_FINGER_TIP,
                                     mp_hands.HandLandmark.PINKY_TIP],
                                    [mp_hands.HandLandmark.MIDDLE_FINGER_DIP,
                                     mp_hands.HandLandmark.RING_FINGER_DIP,
                                     mp_hands.HandLandmark.PINKY_DIP]))):
            return True
        return False

    # Initialize components
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    pygame.mixer.init()

    # Load audio files
    angry_song = 'C:/Users/moham/OneDrive/Desktop/Emorec/mini/predict/static/predict/musics/angry.mp3'
    disgust_song = 'C:/Users/moham/OneDrive/Desktop/Emorec/mini/predict/static/predict/musics/disgust.mp3'
    fear_song = 'C:/Users/moham/OneDrive/Desktop/Emorec/mini/predict/static/predict/musics/fear.mp3'
    happy_song = 'C:/Users/moham/OneDrive/Desktop/Emorec/mini/predict/static/predict/musics/happy.mp3'
    sad_song = 'C:/Users/moham/OneDrive/Desktop/Emorec/mini/predict/static/predict/musics/sad.mp3'
    surprise_song = 'C:/Users/moham/OneDrive/Desktop/Emorec/mini/predict/static/predict/musics/suprise.mp3'
    neutral_song = 'C:/Users/moham/OneDrive/Desktop/Emorec/mini/predict/static/predict/musics/neutral.mp3'

    # Start capturing video
    cap = cv2.VideoCapture(0)

    # Variables to track stable emotion detection for each face
    emotion_queues = {}  # Dictionary to hold queues for each detected face
    stable_emotions = {}  # Dictionary to hold stable emotions for each detected face

    # Variables for hand gesture detection
    hand_closed = False
    index_finger_pointing = False
    audio_playing = False

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture image")
            break

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Analyze each detected face
        for i, (x, y, w, h) in enumerate(faces):
            face_id = f"face_{i}"
            # Extract the face ROI (Region of Interest)
            face_roi = frame[y:y + h, x:x + w]

            # Convert the face ROI to RGB as DeepFace expects RGB images
            face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

            # Analyze the face ROI to predict emotions
            try:
                analysis_list = DeepFace.analyze(img_path=face_roi_rgb, actions=['emotion'], enforce_detection=False)
                analysis = analysis_list[0]  # Get the first (and only) result
                detected_emotion = analysis['dominant_emotion']
            except Exception as e:
                print(f"Error analyzing face ROI: {e}")
                detected_emotion = "Error"

            # Initialize emotion queue if not present
            if face_id not in emotion_queues:
                emotion_queues[face_id] = deque(maxlen=10)

            # Add the detected emotion to the queue
            emotion_queues[face_id].append(detected_emotion)

            # Determine the most common emotion in the queue
            stable_emotion = max(set(emotion_queues[face_id]), key=emotion_queues[face_id].count)
            stable_emotions[face_id] = stable_emotion

            # Draw rectangle around face and label with stable emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, stable_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands
        result = hands.process(frame_rgb)

        # Draw hand landmarks on the frame if hands are detected
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Check if the hand is closed or if only the index finger is pointing
                hand_closed = is_hand_closed(hand_landmarks)
                index_finger_pointing = is_index_finger_pointing(hand_landmarks)

                if hand_closed and not audio_playing:
                    play_sound(stable_emotion)  # Play song based on detected emotion
                    audio_playing = True
                elif not hand_closed and audio_playing:
                    stop_sound()
                    audio_playing = False

                if index_finger_pointing:
                    for (x, y, w, h) in faces:
                        face_id = f"face_{i}"
                        stable_emotion = stable_emotions.get(face_id, "neutral")
                        if stable_emotion == 'happy':
                            show_quote(frame, "Stay positive, stay happy!", x, y)
                        elif stable_emotion == 'sad':
                            show_quote(frame, "It's okay to feel sad sometimes.", x, y)
                        elif stable_emotion == 'neutral':
                            show_quote(frame, "Keep calm and carry on.", x, y)
                        elif stable_emotion == 'angry':
                            show_quote(frame, "Don't let anger control you.", x, y)
                        elif stable_emotion == 'disgust':
                            show_quote(frame, "Choose kindness over disgust.", x, y)
                        elif stable_emotion == 'fear':
                            show_quote(frame, "Face your fears with courage.", x, y)
                        elif stable_emotion == 'surprise':
                            show_quote(frame, "Expect the unexpected!", x, y)

        # Display the video frame
        cv2.imshow('Emotion and Hand Gesture Detection', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


