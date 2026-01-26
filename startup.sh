#!/bin/bash
# Simple Railway startup script

# Set environment
export DJANGO_SETTINGS_MODULE=mini.settings_prod
export PYTHONPATH=/app

# Echo for debugging
echo "Starting application..."
echo "PORT: $PORT"
echo "Settings: $DJANGO_SETTINGS_MODULE"

# Run migrations quietly
python manage.py migrate --noinput >/dev/null 2>&1

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