"""
WSGI config for mini project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/wsgi/
"""

import os
import sys
import logging

from django.core.wsgi import get_wsgi_application

# Use production settings by default, but allow override
settings_module = os.environ.get('DJANGO_SETTINGS_MODULE', 'mini.settings_prod')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', settings_module)

application = get_wsgi_application()

# Pre-load emotion models on startup to avoid first-request delays/failures
logger = logging.getLogger('autism_emotion')
try:
    logger.info("[WSGI STARTUP] Pre-loading emotion models...")
    from predict.AutismEmoRec import get_emotion_models
    get_emotion_models()
    logger.info("[WSGI STARTUP] ✓ Emotion models pre-loaded successfully")
except Exception as e:
    logger.error(f"[WSGI STARTUP] ✗ Failed to pre-load models: {e}")
    # Don't fail startup - fallback mechanisms will handle it
    pass

app = application
