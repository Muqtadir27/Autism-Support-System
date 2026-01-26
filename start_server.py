#!/usr/bin/env python
"""
Simple startup script for Railway deployment
"""

import os
import sys
from django.core.wsgi import get_wsgi_application

# Set the Django settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mini.settings_prod')

if __name__ == "__main__":
    # Import and run gunicorn programmatically
    try:
        from gunicorn.app.wsgiapp import run
        sys.argv = [
            "gunicorn",
            "mini.wsgi:application",
            "--bind", f"0.0.0.0:{os.environ.get('PORT', '8000')}",
            "--workers", "2",
            "--timeout", "120"
        ]
        run()
    except ImportError:
        print("Gunicorn not available, trying to run Django development server")
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mini.settings_prod')
        import django
        from django.core.management import execute_from_command_line
        django.setup()
        
        # Use Django's development server as fallback
        sys.argv = [
            sys.argv[0],
            "runserver",
            f"0.0.0.0:{os.environ.get('PORT', '8000')}"
        ]
        execute_from_command_line(sys.argv)