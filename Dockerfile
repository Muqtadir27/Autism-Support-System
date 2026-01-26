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

# Set environment variable to use different static files storage during build
ENV CONTAINER_BUILD=1
RUN python manage.py collectstatic --noinput --settings=mini.settings

# Make startup script executable
RUN chmod +x startup.sh

# Expose the port the app runs on
EXPOSE $PORT

# Run the application
CMD ["./startup.sh"]