# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install OpenCV with contrib modules specifically for the project
RUN pip install opencv-contrib-python==4.9.0.80

# Copy the rest of the application code
COPY . .

# Make sure static files are collected
RUN python manage.py collectstatic --noinput

# Expose the port the app runs on
EXPOSE $PORT

# Run the application
CMD gunicorn mini.wsgi:application --bind 0.0.0.0:$PORT