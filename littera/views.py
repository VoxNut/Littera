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

IMG_HEIGHT, IMG_WIDTH = 32, 128
CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
ID_TO_CHAR = {i: c for i, c in enumerate(CHARSET)}


def _load_ocr_model():
    global _model
    if _model is None:
        model_path = getattr(settings, 'OCR_MODEL_PATH', os.path.join(
            settings.BASE_DIR, 'models', 'crnn_ocr_ctc_1m_checkpoint.h5'))
        
        # Định nghĩa ctc_loss_layer để load checkpoint
        def ctc_loss_layer(args):
            y_true, y_pred, in_len, lab_len = args
            return keras.backend.ctc_batch_cost(y_true, y_pred, in_len, lab_len)
        
        # Load model với custom_objects
        _model = keras.models.load_model(
            model_path, 
            custom_objects={'ctc_loss_layer': ctc_loss_layer},
            compile=False
        )
        
        # Nếu model có nhiều inputs (train_model), extract base_model
        if len(_model.inputs) > 1:
            # Tìm input 'image'
            image_input = None
            for inp in _model.inputs:
                if 'image' in inp.name:
                    image_input = inp
                    break
            
            # Tìm Dense layer cuối (output của base_model)
            base_output = None
            for layer in reversed(_model.layers):
                if isinstance(layer, keras.layers.Dense):
                    try:
                        shape = layer.output.shape
                        if shape[-1] == len(CHARSET) + 1:  # 63 = 62 chars + blank
                            base_output = layer.output
                            break
                    except:
                        continue
            
            if base_output is not None and image_input is not None:
                _model = keras.Model(inputs=image_input, outputs=base_output)
    
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
