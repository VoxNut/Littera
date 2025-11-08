# Create your views here.
from django.shortcuts import render
from django.conf import settings

from .forms import OCRForm
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

_model = None

# Model supports uppercase + lowercase + digits
IMG_HEIGHT, IMG_WIDTH = 32, 128
CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
ID_TO_CHAR = {i: c for i, c in enumerate(CHARSET)}


def _load_ocr_model():
    global _model
    if _model is None:
        model_path = getattr(settings, 'OCR_MODEL_PATH', os.path.join(
            settings.BASE_DIR, 'models', 'crnn_ocr_ctc_colab_500k.h5'))
        _model = keras.models.load_model(model_path, compile=False)
    return _model


def _preprocess_image(file_obj):
    # Giống pipeline trong notebook: convert('L'), resize cứng (W=128,H=32), scale 0..1
    img = Image.open(file_obj).convert('L').resize(
        (IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
    x = np.array(img, dtype=np.float32) / 255.0
    x = x[..., None]               # (H, W, 1)
    x = np.expand_dims(x, 0)       # (1, H, W, 1)
    return x


def _ctc_decode(y_pred):
    # y_pred shape: (B, T, C+1). T phụ thuộc kiến trúc (notebook là 31)
    B, T = y_pred.shape[0], y_pred.shape[1]
    input_len = np.full((B,), T, dtype=np.int32)
    decoded, _ = keras.backend.ctc_decode(
        y_pred, input_length=input_len, greedy=True)
    seq = decoded[0].numpy()  # (B, Lmax), pad = -1
    texts = []
    for row in seq:
        chars = [ID_TO_CHAR[i] for i in row if i != -1]
        texts.append(''.join(chars))
    return texts


def ocr_view(request):
    if request.method == 'POST':
        form = OCRForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                img_file = form.cleaned_data['image']
                x = _preprocess_image(img_file)
                model = _load_ocr_model()
                y = model.predict(x)
                texts = _ctc_decode(y)
                text = texts[0] if texts else ""
                return render(request, 'littera/ocr.html', {'form': form, 'text': text})
            except Exception as e:
                return render(request, 'littera/ocr.html', {'form': form, 'error': str(e)})
    else:
        form = OCRForm()
    return render(request, 'littera/ocr.html', {'form': form})
