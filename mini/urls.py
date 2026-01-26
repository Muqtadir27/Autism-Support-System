"""
URL configuration for mini project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from core.views import home
from about.views import about,download_pdf
from team.views import team
from contact.views import contact
# Lazy imports to avoid OpenCV startup issues
from django.utils.module_loading import import_string

def predict(request):
    view_func = import_string('predict.views.predict')
    return view_func(request)

def run_detection(request):
    view_func = import_string('predict.views.run_detection')
    return view_func(request)

def download_emotion_log(request):
    view_func = import_string('predict.views.download_emotion_log')
    return view_func(request)

def emotion_log_dashboard(request):
    view_func = import_string('predict.views.emotion_log_dashboard')
    return view_func(request)
urlpatterns = [
    path('admin/', admin.site.urls),
    path('',home,name='Home'),
    path('about/',about,name='About'),
    path('download_pdf/', download_pdf, name='download'),
    path('team/',team,name='Team'),
    path('contact/',contact,name='Contact'),
    path('predict/',predict,name="Predict"),
    path('download_emotion_log/', download_emotion_log, name='download_emotion_log'),
    path('Emotion_Detection_Hand/', run_detection, name='run_detection'),
    path('log_dashboard/', emotion_log_dashboard, name='log_dashboard'),
]

