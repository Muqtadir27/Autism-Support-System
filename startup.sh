#!/bin/bash
# Startup script for Railway deployment

# Set the DJANGO_SETTINGS_MODULE environment variable
export DJANGO_SETTINGS_MODULE=mini.settings_prod

# Get the port from the environment or default to 8000
PORT=${PORT:-8000}

# Start the application using gunicorn
exec gunicorn mini.wsgi:application --bind 0.0.0.0:$PORT --workers 2 --timeout 120 --max-requests 1000 --max-requests-jitter 100