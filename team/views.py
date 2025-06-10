from django.shortcuts import render

# Create your views here.
def team(req):
    return render(req,'team.html')