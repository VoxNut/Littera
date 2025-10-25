

# Create your views here.
from django.shortcuts import render
from django.conf import settings
import easyocr
import cv2
import os
from PIL import Image


def ocr_view(request):
    text = ""
    if request.method == 'POST':
        image = request.FILES['image']
        image_path = os.path.join(settings.MEDIA_ROOT, image.name)

        # Lưu ảnh vào thư mục media/
        with open(image_path, 'wb+') as f:
            for chunk in image.chunks():
                f.write(chunk)

        # Dùng OpenCV để tiền xử lý ảnh
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(image_path, gray)

        # Nhận dạng chữ bằng EasyOCR
        reader = easyocr.Reader(['en', 'vi'], gpu=False)
        result = reader.readtext(image_path, detail=0)
        text = "\n".join(result)

    return render(request, 'littera/upload.html', {'text': text})
