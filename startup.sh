#!/bin/bash
# Startup script for Railway deployment
set -e  # Exit on any error

# Set the DJANGO_SETTINGS_MODULE environment variable
export DJANGO_SETTINGS_MODULE=mini.settings_prod

# Run Django migrations
python manage.py migrate --noinput

# Collect static files again (in case of any new ones)
python manage.py collectstatic --noinput

# Get the port from the environment or default to 8000
PORT=${PORT:-8000}

# Start the application using gunicorn
echo "Starting server on port $PORT..."
exec gunicorn mini.wsgi:application --bind 0.0.0.0:$PORT --workers 2 --timeout 120 --max-requests 1000 --max-requests-jitter 100