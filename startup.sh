#!/bin/bash
# Startup script for Railway deployment
set -e  # Exit on any error

# Set the DJANGO_SETTINGS_MODULE environment variable
export DJANGO_SETTINGS_MODULE=mini.settings_prod

# Print environment info for debugging
echo "PORT: $PORT"
echo "Settings: $DJANGO_SETTINGS_MODULE"

# Run Django migrations
python manage.py migrate --noinput

# Collect static files again (in case of any new ones)
python manage.py collectstatic --noinput

# Get the port from the environment or default to 8000
PORT=${PORT:-8000}

# Validate that the port is numeric
if ! [[ "$PORT" =~ ^[0-9]+$ ]] ; then
   echo "Error: PORT is not a valid number: $PORT" >&2
   exit 1
fi

echo "Validated port: $PORT"

# Start the application using gunicorn with proper timeout settings
echo "Starting server on 0.0.0.0:$PORT..."
exec gunicorn mini.wsgi:application --bind 0.0.0.0:$PORT --workers 1 --timeout 300 --keep-alive 5 --max-requests 1000 --max-requests-jitter 100 --preload --log-level info --access-logfile - --error-logfile -