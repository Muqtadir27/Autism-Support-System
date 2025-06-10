from django.shortcuts import render

# Create your views here.
def contact(req):
    return render(req,'contact.html')