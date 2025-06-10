from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse,FileResponse,Http404
import os
import threading
from .AutismEmoRec import Autism_emotion_recognition
from .facehand import emotion_and_gesture_detection
from .Emogame import emotion_flashcard_game
def predict(request):
    return render(request, 'predict.html')

def run_detection(request):
    if request.method == 'POST' and 'emotion_and_gesture_button' in request.POST:
        # Start emotion and gesture detection in a new thread
        thread = threading.Thread(target=emotion_and_gesture_detection)
        thread.start()
        
        context = {
            'status': 'Detection started',
            'note': 'Please wait for 5 minutes.',
            'instructions': [
                '1. Face the camera.',
                '2. Keep your hand in front of the camera.',
                '3. Close your hand to play a song and open your hand to stop.',
                '4. Show your index finger to display a quote (only the index finger should be extended).'
            ]
        }
        return render(request, 'detection_status.html', context)
    elif request.method == 'POST' and 'autism_emotion_recognition_button' in request.POST:
        # Start Autism emotion recognition in a new thread
        thread = threading.Thread(target=Autism_emotion_recognition)
        thread.start()
        
        context = {
            'status': 'Detection started',
            'note': 'Please Wait !!!!!!!',
            'instructions': [
                'Face the camera.',
                'Download the Emotion log file from below',
            ]
        }
        return render(request, 'Autism.html', context)
    elif request.method == 'POST' and 'Emotion_flashcard_game_buuton' in request.POST:
        # Start Autism emotion recognition in a new thread
        thread = threading.Thread(target=emotion_flashcard_game)
        thread.start()
        
        context = {
            'status': 'Detection started',
            'note': 'Please Wait !!!!!!!',
            'instructions': [
                'Face the camera.',
            ]
        }
        return render(request, 'Game.html', context)
    else:
        return JsonResponse({'status': 'Error', 'message': 'Invalid request method or missing button identifier.'}, status=400)
    

def download_emotion_log(request):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(current_dir, "emotion_log.xlsx")

    if os.path.exists(log_file):
        return FileResponse(open(log_file, 'rb'), as_attachment=True, filename='emotion_log.xlsx')
    else:
        raise Http404("Log file not found")