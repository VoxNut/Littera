# Create your views here.
from django.shortcuts import render
from django.conf import settings

from .forms import OCRForm
import os
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow import keras
import io

_model = None

IMG_HEIGHT, IMG_WIDTH = 32, 128
# Original CHARSET without space (model was trained on this)
CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
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


def _preprocess_word_image(img):
    """Preprocess a word region (numpy array) for OCR"""
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Convert to PIL for resize
    pil_img = Image.fromarray(img).resize(
        (IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
    
    x = np.array(pil_img, dtype=np.float32) / 255.0
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


def _detect_words_spacing(image_array):
    """Detect words by finding text lines and analyzing character spacing"""
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # First, detect text lines using horizontal dilation
    line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
    dilated_lines = cv2.dilate(thresh, line_kernel, iterations=1)
    
    # Find line contours
    line_contours, _ = cv2.findContours(dilated_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    all_word_boxes = []
    
    for line_contour in line_contours:
        x, y, w, h = cv2.boundingRect(line_contour)
        
        # Extract line region
        line_roi = thresh[y:y+h, x:x+w]
        
        # Find character contours in this line
        char_contours, _ = cv2.findContours(line_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get bounding boxes for characters
        char_boxes = []
        for char_contour in char_contours:
            cx, cy, cw, ch = cv2.boundingRect(char_contour)
            if cw > 3 and ch > 5:  # Filter noise
                char_boxes.append((x + cx, y + cy, x + cx + cw, y + cy + ch))
        
        if not char_boxes:
            continue
        
        # Sort characters left to right
        char_boxes = sorted(char_boxes, key=lambda b: b[0])
        
        # Group characters into words by analyzing spacing
        word_groups = []
        current_word = [char_boxes[0]]
        
        for i in range(1, len(char_boxes)):
            prev_box = char_boxes[i-1]
            curr_box = char_boxes[i]
            
            # Calculate gap between characters
            gap = curr_box[0] - prev_box[2]
            
            # Calculate average character width
            avg_width = (prev_box[2] - prev_box[0] + curr_box[2] - curr_box[0]) / 2
            
            # If gap is larger than 50% of character width, it's a word boundary
            if gap > avg_width * 0.5:
                # Save current word
                word_groups.append(current_word)
                current_word = [curr_box]
            else:
                current_word.append(curr_box)
        
        # Don't forget last word
        word_groups.append(current_word)
        
        # Create bounding box for each word group
        for word_group in word_groups:
            min_x = min(box[0] for box in word_group)
            min_y = min(box[1] for box in word_group)
            max_x = max(box[2] for box in word_group)
            max_y = max(box[3] for box in word_group)
            all_word_boxes.append((min_x, min_y, max_x, max_y))
    
    return all_word_boxes


def _detect_words_contour(image_array):
    """Detect words using contour detection - better for small text"""
    # Convert to grayscale
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to get binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Calculate average character width to determine appropriate kernel size
    # Find individual character contours first
    char_contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if char_contours:
        # Get average character width
        char_widths = [cv2.boundingRect(c)[2] for c in char_contours if cv2.boundingRect(c)[2] > 5]
        if char_widths:
            avg_char_width = np.mean(char_widths)
            # Use kernel width proportional to character width (about 30-40% of avg char width)
            kernel_width = max(3, int(avg_char_width * 0.35))
        else:
            kernel_width = 8
    else:
        kernel_width = 8
    
    # Apply morphological operations to connect characters in words
    # Use adaptive horizontal kernel based on character size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    # Find contours on dilated image (will find word-level regions)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get bounding boxes for each contour
    word_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter out very small regions (noise) and very thin regions
        if w > 10 and h > 8:
            word_boxes.append((x, y, x+w, y+h))
    
    # Sort word boxes left to right
    word_boxes = sorted(word_boxes, key=lambda b: b[0])
    
    return word_boxes


def _detect_words_east(image_array, east_model_path=None):
    """Detect word bounding boxes using EAST detector"""
    if east_model_path is None:
        east_model_path = getattr(settings, 'EAST_MODEL_PATH', 
                                 os.path.join(settings.BASE_DIR, 'models', 'frozen_east_text_detection.pb'))
    
    if not os.path.exists(east_model_path):
        return None, []
    
    orig_h, orig_w = image_array.shape[:2]
    
    # EAST requires dimensions to be multiples of 32
    # Use larger size for better detection of small text
    new_w, new_h = 640, 640
    ratio_w = orig_w / new_w
    ratio_h = orig_h / new_h
    
    # Resize for EAST
    resized = cv2.resize(image_array, (new_w, new_h))
    
    # Load EAST model
    net = cv2.dnn.readNet(east_model_path)
    
    # Prepare blob
    blob = cv2.dnn.blobFromImage(resized, 1.0, (new_w, new_h),
                                 (123.68, 116.78, 103.94),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    
    # Get predictions
    (scores, geometry) = net.forward(['feature_fusion/Conv_7/Sigmoid',
                                     'feature_fusion/concat_3'])
    
    # Decode detections with lower confidence threshold
    boxes = _decode_predictions(scores, geometry, min_confidence=0.3)
    
    # Convert boxes to original image coordinates
    word_boxes = []
    for (startX, startY, endX, endY) in boxes:
        startX = int(startX * ratio_w)
        startY = int(startY * ratio_h)
        endX = int(endX * ratio_w)
        endY = int(endY * ratio_h)
        word_boxes.append((startX, startY, endX, endY))
    
    # Sort boxes left to right
    word_boxes = sorted(word_boxes, key=lambda b: b[0])
    
    return image_array, word_boxes


def _decode_predictions(scores, geometry, min_confidence=0.3):
    """Decode EAST detector predictions"""
    (numRows, numCols) = scores.shape[2:4]
    boxes = []
    confidences = []
    
    for y in range(numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        
        for x in range(numCols):
            if scoresData[x] < min_confidence:
                continue
            
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            
            boxes.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
    
    # Apply non-maximum suppression with lower threshold
    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.3)
        if len(indices) > 0:
            return [boxes[i] for i in indices.flatten()]
    
    return []


def _group_words_into_lines(word_boxes):
    """Group word boxes into lines based on Y-coordinate"""
    if not word_boxes:
        return []
    
    # Sort by Y coordinate first
    sorted_boxes = sorted(word_boxes, key=lambda b: b[1])
    
    lines = []
    current_line = [sorted_boxes[0]]
    
    for i in range(1, len(sorted_boxes)):
        prev_box = sorted_boxes[i-1]
        curr_box = sorted_boxes[i]
        
        # Calculate vertical distance
        prev_center_y = (prev_box[1] + prev_box[3]) / 2
        curr_center_y = (curr_box[1] + curr_box[3]) / 2
        avg_height = ((prev_box[3] - prev_box[1]) + (curr_box[3] - curr_box[1])) / 2
        
        # If vertical distance is less than half the average height, same line
        if abs(curr_center_y - prev_center_y) < avg_height * 0.5:
            current_line.append(curr_box)
        else:
            # New line - save current line and start new one
            # Sort current line left to right
            current_line = sorted(current_line, key=lambda b: b[0])
            lines.append(current_line)
            current_line = [curr_box]
    
    # Don't forget the last line
    current_line = sorted(current_line, key=lambda b: b[0])
    lines.append(current_line)
    
    return lines


def home_view(request):
    """Render site home page"""
    return render(request, 'littera/home.html')


def ocr_view(request):
    if request.method == 'POST':
        form = OCRForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                img_file = form.cleaned_data['image']
                
                # Check if word detection is enabled (default: ON for multi-word support)
                detect_words = request.POST.get('detect_words', 'on') == 'on'
                
                model = _load_ocr_model()
                
                if detect_words:
                    # Read image as numpy array for word detection
                    img_bytes = img_file.read()
                    img_file.seek(0)  # Reset file pointer
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    image_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    # Try spacing-based detection first (most accurate for word boundaries)
                    word_boxes = _detect_words_spacing(image_array)
                    
                    if not word_boxes or len(word_boxes) <= 1:
                        # Try EAST as fallback
                        _, word_boxes = _detect_words_east(image_array)
                    
                    if word_boxes and len(word_boxes) > 1:
                        # Multiple words detected - group into lines and OCR
                        lines = _group_words_into_lines(word_boxes)
                        
                        recognized_lines = []
                        for line_boxes in lines:
                            recognized_words = []
                            for (startX, startY, endX, endY) in line_boxes:
                                # Extract word region with padding
                                padding = 5
                                startY = max(0, startY - padding)
                                endY = min(image_array.shape[0], endY + padding)
                                startX = max(0, startX - padding)
                                endX = min(image_array.shape[1], endX + padding)
                                
                                word_img = image_array[startY:endY, startX:endX]
                                
                                if word_img.size == 0:
                                    continue
                                
                                # Preprocess and recognize
                                x = _preprocess_word_image(word_img)
                                y = model.predict(x, verbose=0)
                                texts = _ctc_decode(y)
                                word_text = texts[0] if texts else ""
                                
                                if word_text:
                                    recognized_words.append(word_text)
                            
                            if recognized_words:
                                recognized_lines.append(' '.join(recognized_words))
                        
                        text = '\n'.join(recognized_lines)
                    else:
                        # Single word or no detection - process whole image
                        x = _preprocess_image(img_file)
                        y = model.predict(x, verbose=0)
                        texts = _ctc_decode(y)
                        text = texts[0] if texts else ""
                else:
                    # Standard single-word OCR (no word detection)
                    x = _preprocess_image(img_file)
                    y = model.predict(x)
                    texts = _ctc_decode(y)
                    text = texts[0] if texts else ""
                
                return render(request, 'littera/ocr.html', {
                    'form': form, 
                    'text': text,
                    'detect_words': detect_words
                })
            except Exception as e:
                return render(request, 'littera/ocr.html', {
                    'form': form, 
                    'error': str(e)
                })
    else:
        form = OCRForm()
    return render(request, 'littera/ocr.html', {'form': form, 'detect_words': True})

def home_view(request):
    return render(request, 'littera/home.html')

def login_view(request):
    return render(request, 'littera/login.html')