#!/bin/bash
# Railway startup script with emotion detection model preloading

# Set environment
export DJANGO_SETTINGS_MODULE=mini.settings_prod
export PYTHONPATH=/app
export DEEPFACE_HOME=/tmp/deepface
export TF_CPP_MIN_LOG_LEVEL=2

# Echo for debugging
echo "Starting application..."
echo "PORT: $PORT"
echo "Settings: $DJANGO_SETTINGS_MODULE"

# Run migrations quietly
python manage.py migrate --noinput >/dev/null 2>&1

# Pre-load emotion detection models (critical for production)
echo "Pre-loading emotion detection models..."
python -c "
import sys
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mini.settings_prod')
import django
django.setup()
try:
    from predict.AutismEmoRec import get_emotion_models
    print('Loading TensorFlow models...')
    models = get_emotion_models()
    print('✓ TensorFlow models loaded successfully')
except Exception as e:
    print(f'Warning: Could not pre-load models: {e}')
    print('Will attempt to load on first request')
" 2>&1 || true

# Start server immediately
echo "Starting Gunicorn on port $PORT"
exec gunicorn mini.wsgi:application \
    --bind 0.0.0.0:$PORT \
    --workers 1 \
    --timeout 60 \
    --keep-alive 2 \
    --max-requests 100 \
    --preload \
    --log-level warning