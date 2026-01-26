#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone script to run emotion recognition in a separate process.
This allows OpenCV windows to display properly on Windows.
"""
import sys
import os
import logging

# Set up logging to file
log_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_file = os.path.join(log_dir, 'emotion_recognition.log')
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    logging.info("Starting emotion recognition script...")
    logging.info(f"Python executable: {sys.executable}")
    logging.info(f"Working directory: {os.getcwd()}")
    logging.info(f"Script path: {__file__}")
    
    try:
        logging.info("Importing Autism_emotion_recognition...")
        from predict.AutismEmoRec import Autism_emotion_recognition
        logging.info("Import successful. Starting emotion recognition...")
        Autism_emotion_recognition()
        logging.info("Emotion recognition completed.")
    except Exception as e:
        error_msg = f"Error running emotion recognition: {e}"
        logging.error(error_msg, exc_info=True)
        print(error_msg)
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
