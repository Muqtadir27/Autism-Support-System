# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for OpenCV and other packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libglib2.0-dev \
        libgtk2.0-dev \
        pkg-config \
        libgl1-mesa-glx \
        libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Make startup script executable
RUN chmod +x startup.sh

# Set environment variable to use different static files storage during build
ENV CONTAINER_BUILD=1
RUN python manage.py collectstatic --noinput --settings=mini.settings

# Expose the port the app runs on
EXPOSE $PORT

# Run the application
CMD ["./startup.sh"]