from django.shortcuts import render
from django.core.mail import send_mail
from django.contrib import messages
from django.conf import settings

def contact(req):
    if req.method == 'POST':
        name = req.POST.get('name')
        email = req.POST.get('email')
        message = req.POST.get('message')
        
        # Send Email
        subject = f"NEURAL-CONTACT: New Submission from {name}"
        email_message = f"NAME: {name}\nEMAIL: {email}\n\nDATA:\n{message}"
        
        try:
            send_mail(
                subject,
                email_message,
                'webmaster@localhost',
                ['muqtadir27@gmail.com'],
                fail_silently=False,
            )
            messages.success(req, 'TRANSMISSION SUCCESSFUL: Your message has been sent.')
        except Exception as e:
            print(f"Error sending email: {e}")
            messages.error(req, 'TRANSMISSION ERROR: Unable to send your message.')
            
    return render(req, 'contact.html')