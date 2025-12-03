from django.urls import path
from .views import home_view
from .views import ocr_view
from .views import login_view

urlpatterns = [
    path('ocr/', ocr_view, name='ocr'),
    path('', home_view, name='home'),
]
