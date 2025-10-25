from django.contrib import admin
from django.urls import path
from ocr_app.views import ocr_view
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', ocr_view, name='ocr'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
