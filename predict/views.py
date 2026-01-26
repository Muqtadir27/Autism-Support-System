from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse,FileResponse,Http404
import os
import threading
import subprocess
import sys
import pandas as pd
# Note: OpenCV imports are deferred to avoid startup issues
# See get_autism_emotion_recognition(), get_emotion_and_gesture_detection(), get_vocal_expression_interpretation() functions
# Import functions (these will be imported when needed to avoid startup issues)
def get_autism_emotion_recognition():
    from .AutismEmoRec import Autism_emotion_recognition
    return Autism_emotion_recognition

def get_emotion_and_gesture_detection():
    from .facehand import emotion_and_gesture_detection
    return emotion_and_gesture_detection

def get_vocal_expression_interpretation():
    from .vocal_emo import vocal_expression_interpretation
    return vocal_expression_interpretation

def predict(request):
    return render(request, 'predict.html')

def run_detection(request):
    if request.method == 'POST' and 'emotion_and_gesture_button' in request.POST:
        # Vocal Expression Interpretation
        vocal_func = get_vocal_expression_interpretation()
        thread = threading.Thread(target=vocal_func)
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
        # Redirect to web-based camera interface (no threading needed)
        
        from django.template.context_processors import csrf
        context = {
            'status': 'Web-based emotion detection ready',
            'note': 'Position your face in front of your device camera for emotion detection.',
            'instructions': [
                'Click START CAMERA to begin',
                'Allow camera access when prompted',
                'Position your face within the camera view',
                'Emotions will be detected and displayed in real-time',
                'Click STOP CAMERA when finished',
                'Download the Emotion log file when done'
            ]
        }
        context.update(csrf(request))  # Add CSRF token to context
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

def process_camera_frame(request):
    print(f"Received request to process camera frame")
    if request.method == 'POST':
        try:
            # Get the uploaded image
            image_file = request.FILES.get('image')
            if not image_file:
                print("No image file received")
                return JsonResponse({'success': False, 'error': 'No image provided'})
            
            print(f"Received image file: {image_file.name}, size: {image_file.size} bytes")
            
            # Process the image to detect emotion
            # Import the function here to avoid startup issues
            from .AutismEmoRec import process_single_frame_for_emotion
            
            # Call the emotion detection function
            emotion = process_single_frame_for_emotion(image_file)
            print(f"Detected emotion: {emotion}")
            
            return JsonResponse({'success': True, 'emotion': emotion})
        except Exception as e:
            print(f"Error processing camera frame: {e}")
            import traceback
            traceback.print_exc()
            return JsonResponse({'success': False, 'error': str(e)})
    else:
        print("Received non-POST request")
        return JsonResponse({'success': False, 'error': 'Only POST requests allowed'})