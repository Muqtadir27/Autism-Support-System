import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
from deepface import DeepFace
import pygame
import random
import os

def emotion_flashcard_game():
    # List of stories with associated questions and answers
    stories = [
        {
            'story': "John is at a birthday party, surrounded by friends and family. How does John feel?",
            'question': 'Happy'
        },
        {
            'story': "Emily just missed her flight for an important business meeting. How is Emily feeling?",
            'question': 'Sad'
        },
        {
            'story': "Michael is about to give a speech in front of a large audience. What emotion is Michael experiencing?",
            'question': 'Fear'
        },
        {
            'story': "Sophia won the first prize in a drawing competition. How does Sophia feel?",
            'question': 'Surprised'
        },
        {
            'story': "David missed his train and will be late for an important interview. How is David feeling?",
            'question': 'Sad'
        },
        {
            'story': "Emma is watching a horror movie alone at night. What emotion might Emma be experiencing?",
            'question': 'Fear'
        },
        {
            'story': "Jack just got promoted at work. How does Jack feel?",
            'question': 'Happy'
        },
        {
            'story': "Sarah is about to meet her favorite celebrity in person. What emotion is Sarah experiencing?",
            'question': 'Happy'
        },
        {
            'story': "Tom is about to bungee jump from a tall bridge. What emotion might Tom be experiencing?",
            'question': 'Fear'
        },
        {
            'story': "Lily is tasting her favorite dessert after a long time. How does Lily feel?",
            'question': 'Happy'
        }
    ]

    # Define sound files for positive and negative feedback
    CORRECT_SOUND = 'sounds/celebrate.wav'
    INCORRECT_SOUND = 'sounds/oops.wav'

    def start_video():
        nonlocal video_capture, frame_id
        start_button.config(state=tk.DISABLED)
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            messagebox.showerror("Error", "Failed to open video capture device.")
            root.quit()

        show_frame()

    def show_frame():
        nonlocal frame_id
        ret, frame = video_capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect emotions and draw rectangle around face with detected emotion
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            for face in result:
                x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
                emotion = face['dominant_emotion']
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            image = Image.fromarray(frame)
            image = ImageTk.PhotoImage(image)

            video_label.config(image=image)
            video_label.image = image
            frame_id = root.after(10, show_frame)
        else:
            messagebox.showerror("Error", "Failed to capture video frame.")
            root.quit()

    def detect_emotion(frame):
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            if len(result) > 0:
                detected_emotion = result[0]['dominant_emotion']
                return detected_emotion
            return None
        except Exception as e:
            print(f"Error analyzing emotion: {e}")
            return None

    def capture_emotion():
        nonlocal video_capture
        ret, frame = video_capture.read()
        if ret:
            detected_emotion = detect_emotion(frame)
            if detected_emotion:
                check_answer(detected_emotion)
            else:
                messagebox.showerror("Error", "Failed to analyze emotion.")
                start_button.config(state=tk.NORMAL)
        else:
            messagebox.showerror("Error", "Failed to capture video frame.")
            root.quit()

    def start_game():
        nonlocal score, questions_asked, used_stories, current_story, current_question
        score = 0
        questions_asked = 0
        used_stories = []
        update_score()
        next_question()

    def update_score():
        score_label.config(text=f"Score: {score}")

    def next_question():
        nonlocal questions_asked, current_story, current_question, used_stories
        if questions_asked >= 5:
            end_game()
            return

        # Randomly select a story that hasn't been used yet
        available_stories = [story for story in stories if story not in used_stories]
        if not available_stories:
            end_game()
            return

        current_story = random.choice(available_stories)
        used_stories.append(current_story)

        story_text = current_story['story']
        story_label.config(text=story_text)

        current_question = current_story['question']
        questions_asked += 1

        start_button.config(state=tk.NORMAL)

    def check_answer(detected_emotion):
        nonlocal score
        if detected_emotion.lower() == current_question.lower():
            score += 1
            messagebox.showinfo("Correct!", "You identified the emotion correctly.")
            play_sound(CORRECT_SOUND)
            update_score()
            root.after(2000, next_question)  # Smooth transition to the next question after 2 seconds
        else:
            messagebox.showerror("Incorrect", f"Sorry, that's incorrect.")
            play_sound(INCORRECT_SOUND)
            root.quit()

    def end_game():
        if score >= 3:
            messagebox.showinfo("Game Over", f"Your final score: {score}\nYou're making progress!")
        else:
            messagebox.showinfo("Game Over", f"Your final score: {score}\nYou have some things to work on.")
        root.quit()

    def play_sound(sound_file):
        if os.path.exists(sound_file):
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.play()
        else:
            print(f"Sound file {sound_file} not found.")

    # Initialize main game variables
    score = 0
    questions_asked = 0
    used_stories = []
    current_story = None
    current_question = None
    video_capture = None
    frame_id = None

    # Initialize pygame for sound playback
    pygame.mixer.init()

    # Initialize Tkinter root window
    root = tk.Tk()
    root.title("Emotion Flashcard Game")

    flashcard_frame = tk.Frame(root, bg='#ffffff')  # Light background for better readability
    flashcard_frame.pack(fill='both', expand=True)

    controls_frame = tk.Frame(root, bg='#ffffff')
    controls_frame.pack(fill='both', expand=True)

    start_button = tk.Button(controls_frame, text="Start Video", command=start_video, font=('Helvetica', 14), bg='#4CAF50', fg='white')
    start_button.pack(side='left', padx=10, pady=10)

    capture_button = tk.Button(controls_frame, text="Capture", command=capture_emotion, font=('Helvetica', 14), bg='#2196F3', fg='white')
    capture_button.pack(side='left', padx=10, pady=10)

    quit_button = tk.Button(controls_frame, text="Quit", command=root.quit, font=('Helvetica', 14), bg='#f44336', fg='white')
    quit_button.pack(side='right', padx=10, pady=10)

    story_label = tk.Label(flashcard_frame, text="", font=('Helvetica', 16), bg='#ffffff', wraplength=600, justify='left')
    story_label.pack(pady=20, padx=10)

    video_label = tk.Label(flashcard_frame, bg='#ffffff')
    video_label.pack()

    score_label = tk.Label(flashcard_frame, text="Score: 0", font=('Helvetica', 14), bg='#ffffff', fg='#4CAF50')
    score_label.pack(pady=20)

    start_game()

    root.mainloop()

