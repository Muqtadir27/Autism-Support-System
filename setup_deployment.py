#!/usr/bin/env python
"""
Setup script to help with deployment configuration
"""
import os
import sys
import random
import string

def generate_secret_key():
    """Generate a new Django secret key"""
    chars = 'abcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*(-_=+)'
    return ''.join(random.choice(chars) for _ in range(50))

def create_env_file():
    """Create a sample .env file for local development"""
    env_content = f"""# Autism Support System - Environment Variables
DJANGO_SECRET_KEY={generate_secret_key()}
DEBUG=True
EMAIL_BACKEND=django.core.mail.backends.smtp.EmailBackend
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USE_TLS=True
EMAIL_HOST_USER=your-email@gmail.com
EMAIL_HOST_PASSWORD=your-app-password
"""
    
    with open('.env.example', 'w') as f:
        f.write(env_content)
    
    print("Created .env.example file with sample configuration")
    print("Remember to create your own .env file for local development")
    print("and never commit .env files to version control!")

def main():
    print("Autism Support System - Deployment Setup")
    print("="*50)
    
    print("\n1. Generated a new Django secret key:")
    secret_key = generate_secret_key()
    print(secret_key)
    
    print("\n2. Creating .env.example file...")
    create_env_file()
    
    print("\n3. Deployment preparation complete!")
    print("\nNext steps:")
    print("- Fork this repository to your GitHub account")
    print("- Set up your chosen hosting platform (Vercel, Heroku, or Railway)")
    print("- Configure environment variables as described in DEPLOYMENT_GUIDE.md")
    print("- Push your code to GitHub to trigger automatic deployment")
    print("- Visit your deployed application URL")
    
    print("\nFor email notifications, make sure to:")
    print("- Use an email provider that supports SMTP (like Gmail)")
    print("- Generate an app password for authentication")
    print("- Configure the email settings in your hosting platform")

if __name__ == "__main__":
    main()
