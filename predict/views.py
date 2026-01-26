from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse,FileResponse,Http404
import os
import threading
import subprocess
import sys
import pandas as pd
from .AutismEmoRec import Autism_emotion_recognition
from .facehand import emotion_and_gesture_detection
from .vocal_emo import vocal_expression_interpretation

def predict(request):
    return render(request, 'predict.html')

def run_detection(request):
    if request.method == 'POST' and 'emotion_and_gesture_button' in request.POST:
        # Vocal Expression Interpretation
        thread = threading.Thread(target=vocal_expression_interpretation)
        thread.start()
        
        context = {
            'status': 'Vocal Support Active',
            'note': 'The system is interpreting your vocal expressions.',
            'instructions': [
                '1. Ensure your microphone is active.',
                '2. Speak clearly into the microphone.',
                '3. Say "Stop" or "Exit" to terminate the session.',
                '4. The system will provide supportive audio responses.'
            ]
        }
        return render(request, 'vocal_support.html', context)
    elif request.method == 'POST' and 'autism_emotion_recognition_button' in request.POST:
        # Start Autism emotion recognition in a separate process so OpenCV window displays properly
        try:
            # Get the path to the script
            current_dir = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(current_dir, 'run_emotion_recognition.py')
            
            # Verify script exists
            if not os.path.exists(script_path):
                raise FileNotFoundError(f"Script not found: {script_path}")
            
            # Get Python executable from the virtual environment or system
            python_exe = sys.executable
            
            print(f"Starting emotion recognition process...")
            print(f"Python: {python_exe}")
            print(f"Script: {script_path}")
            print(f"Working dir: {os.path.dirname(current_dir)}")
            
            # Start the process - use threading instead (simpler and works better for GUI)
            thread = threading.Thread(target=Autism_emotion_recognition, daemon=False)
            thread.start()
            print(f"Emotion recognition thread started")
        except Exception as e:
            error_msg = f"Error starting emotion recognition process: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
        
        context = {
            'status': 'Detection started',
            'note': 'Camera window should open shortly. Please face the camera.',
            'instructions': [
                'A camera window will open showing your face.',
                'The system will detect your emotion in real-time.',
                'Press "q" in the camera window to stop detection.',
                'Download the Emotion log file from below when done.',
            ]
        }
        return render(request, 'Autism.html', context)
    elif request.method == 'POST' and 'Emotion_flashcard_game_buuton' in request.POST:
        return redirect('log_dashboard')
    else:
        return JsonResponse({'status': 'Error', 'message': 'Invalid request method or missing button identifier.'}, status=400)

def emotion_log_dashboard(request):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(current_dir, "emotion_log.xlsx")
    
    if os.path.exists(log_file):
        df = pd.read_excel(log_file)
        # Basic stats
        emotion_counts = df['emotion'].value_counts().to_dict()
        labels = list(emotion_counts.keys())
        data = list(emotion_counts.values())
        
        # Timeline data (last 20 entries)
        timeline = df.tail(20).to_dict('records')
        
        context = {
            'labels': labels,
            'data': data,
            'timeline': timeline
        }
        return render(request, 'log_dashboard.html', context)
    else:
        return render(request, 'log_dashboard.html', {'error': 'No emotional log data available yet.'})

def download_emotion_log(request):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(current_dir, "emotion_log.xlsx")

    if os.path.exists(log_file):
        return FileResponse(open(log_file, 'rb'), as_attachment=True, filename='emotion_log.xlsx')
    else:
        raise Http404("Log file not found")