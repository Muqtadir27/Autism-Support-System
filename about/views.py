from django.shortcuts import render
from django.http import FileResponse, Http404,HttpResponse
from django.conf import settings
import os
# Create your views here.
def about(req):
    return render(req,'about.html')
def download_pdf(request):
    pdf_file_path = os.path.join(settings.BASE_DIR, 'about\\static\\about\\doc\\MiniDocument.pdf')

    if os.path.exists(pdf_file_path):
        with open(pdf_file_path, 'rb') as pdf_file:
            response = HttpResponse(pdf_file.read(), content_type='application/pdf')
            response['Content-Disposition'] = 'attachment; filename="MiniDocument.pdf"'
            return response
    else:
        raise Http404("PDF file not found")