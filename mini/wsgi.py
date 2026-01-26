"""
WSGI config for mini project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

import os
# Use production settings by default, but allow override
settings_module = os.environ.get('DJANGO_SETTINGS_MODULE', 'mini.settings_prod')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', settings_module)

application = get_wsgi_application()
app=application
