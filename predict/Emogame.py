import tkinter as tk
from tkinter import messagebox
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
        import cv2
        nonlocal video_capture, frame_id
        start_button.config(state=tk.DISABLED)
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            messagebox.showerror("Error", "Failed to open video capture device.")
            root.quit()

        show_frame()

    def show_frame():
        import cv2
        nonlocal frame_id, video_capture, root, video_label
        if video_capture is None:
            return
        ret, frame = video_capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect emotions and draw rectangle around face with detected emotion
            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                for face in result:
                    x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
                    emotion = face['dominant_emotion']
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception as e:
                print(f"DeepFace error in show_frame: {e}")

            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image)

            video_label.config(image=photo)
            video_label.image = photo
            frame_id = root.after(10, show_frame)
        else:
            messagebox.showerror("Error", "Failed to capture video frame.")
            root.quit()

    def detect_emotion(frame):
        import cv2
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
    root.title("NEURAL-EMO INTERFACE")
    root.configure(bg='#0a0e14')
    root.geometry("900x800")

    # Sci-fi Styling
    accent_color = '#00f2ff'
    bg_dark = '#0a0e14'
    text_color = '#e0e6ed'

    flashcard_frame = tk.Frame(root, bg=bg_dark, bd=2, relief='ridge')
    flashcard_frame.pack(fill='both', expand=True, padx=20, pady=10)

    controls_frame = tk.Frame(root, bg=bg_dark)
    controls_frame.pack(fill='both', expand=False, padx=20, pady=20)

    # Style Buttons
    btn_style = {'font': ('Courier', 12, 'bold'), 'bd': 0, 'padx': 20, 'pady': 10, 'cursor': 'hand2'}

    start_button = tk.Button(controls_frame, text="[INITIALIZE SENSORS]", command=start_video, **btn_style, bg='#1a2634', fg=accent_color, activebackground=accent_color, activeforeground=bg_dark)
    start_button.pack(side='left', padx=10)

    capture_button = tk.Button(controls_frame, text="[ANALYZE EMOTION]", command=capture_emotion, **btn_style, bg='#1a2634', fg='#00ff41', activebackground='#00ff41', activeforeground=bg_dark)
    capture_button.pack(side='left', padx=10)

    quit_button = tk.Button(controls_frame, text="[TERMINATE]", command=root.quit, **btn_style, bg='#1a2634', fg='#ff3131', activebackground='#ff3131', activeforeground=bg_dark)
    quit_button.pack(side='right', padx=10)

    # Header and Video Labels
    header_label = tk.Label(flashcard_frame, text="LOGICAL SCENARIO ANALYSIS", font=('Courier', 10), bg=bg_dark, fg=accent_color)
    header_label.pack(pady=(10, 0))

    story_label = tk.Label(flashcard_frame, text="", font=('Courier', 14), bg=bg_dark, fg=text_color, wraplength=800, justify='center', pady=20)
    story_label.pack(pady=10, padx=20)

    video_label = tk.Label(flashcard_frame, bg='#111821', highlightthickness=1, highlightbackground=accent_color)
    video_label.pack(pady=10)

    score_label = tk.Label(flashcard_frame, text="NEURAL ACCURACY: 0%", font=('Courier', 12, 'bold'), bg=bg_dark, fg='#00ff41')
    score_label.pack(pady=10)

    def update_score():
        accuracy = (score / questions_asked * 100) if questions_asked > 0 else 0
        score_label.config(text=f"NEURAL ACCURACY: {accuracy:.1f}%")

    start_game()

    root.mainloop()

