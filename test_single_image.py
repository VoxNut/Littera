"""
Test OCR on a single image with word detection
Usage: python test_single_image.py path/to/image.png [--detect-words]
"""
import sys
import os
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow import keras

IMG_HEIGHT, IMG_WIDTH = 32, 128
# Original CHARSET (model was trained on this, no space)
CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
ID_TO_CHAR = {i: c for i, c in enumerate(CHARSET)}


def ctc_loss_layer(args):
    """Custom layer for loading checkpoint"""
    y_true, y_pred, in_len, lab_len = args
    return keras.backend.ctc_batch_cost(y_true, y_pred, in_len, lab_len)


def load_model(model_path='models/crnn_ocr_ctc_1m_checkpoint.h5'):
    """Load the trained CRNN model"""
    model = keras.models.load_model(
        model_path, 
        custom_objects={'ctc_loss_layer': ctc_loss_layer},
        compile=False
    )
    
    # If train_model with multiple inputs, extract base_model
    if len(model.inputs) > 1:
        image_input = None
        for inp in model.inputs:
            if 'image' in inp.name:
                image_input = inp
                break
        
        base_output = None
        for layer in reversed(model.layers):
            if isinstance(layer, keras.layers.Dense):
                try:
                    shape = layer.output.shape
                    if shape[-1] == len(CHARSET) + 1:
                        base_output = layer.output
                        break
                except:
                    continue
        
        if base_output is not None and image_input is not None:
            model = keras.Model(inputs=image_input, outputs=base_output)
    
    return model


def preprocess_image(image_path):
    """Preprocess image for OCR"""
    img = Image.open(image_path).convert('L').resize(
        (IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
    x = np.array(img, dtype=np.float32) / 255.0
    x = x[..., None]               # (H, W, 1)
    x = np.expand_dims(x, 0)       # (1, H, W, 1)
    return x


def preprocess_word_image(img):
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


def ctc_decode(y_pred):
    """Decode CTC output to text"""
    B, T = y_pred.shape[0], y_pred.shape[1]
    input_len = np.full((B,), T, dtype=np.int32)
    decoded, _ = keras.backend.ctc_decode(
        y_pred, input_length=input_len, greedy=True)
    seq = decoded[0].numpy()
    texts = []
    for row in seq:
        chars = [ID_TO_CHAR[i] for i in row if i != -1]
        texts.append(''.join(chars))
    return texts


def detect_words_spacing(image_path):
    """Detect words by finding text lines and analyzing character spacing"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # First, detect text lines using vertical dilation
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
    
    return image, all_word_boxes


def detect_words_contour(image_path):
    """Detect words using contour detection - better for small text"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
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
    
    return image, word_boxes


def detect_words_east(image_path, east_model_path='models/frozen_east_text_detection.pb'):
    """Detect word bounding boxes using EAST detector"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    orig_h, orig_w = image.shape[:2]
    
    # EAST requires dimensions to be multiples of 32
    # Use larger size for better detection of small text
    new_w, new_h = 640, 640
    ratio_w = orig_w / new_w
    ratio_h = orig_h / new_h
    
    # Resize for EAST
    resized = cv2.resize(image, (new_w, new_h))
    
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
    boxes = decode_predictions(scores, geometry, min_confidence=0.3)
    
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
    
    return image, word_boxes


def decode_predictions(scores, geometry, min_confidence=0.3):
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


def group_words_into_lines(word_boxes):
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


def recognize_text(image_path, model_path='models/crnn_ocr_ctc_1m_checkpoint.h5', detect_words=False):
    """Recognize text from an image"""
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    if detect_words:
        print(f"Detecting words in: {image_path}")
        try:
            # Try spacing-based detection first (most accurate for word boundaries)
            print("Trying spacing-based word detection...")
            image, word_boxes = detect_words_spacing(image_path)
            
            if not word_boxes or len(word_boxes) <= 1:
                # Try EAST as fallback
                print("Spacing found few words, trying EAST detection...")
                image_east, word_boxes_east = detect_words_east(image_path)
                if word_boxes_east and len(word_boxes_east) > len(word_boxes):
                    image = image_east
                    word_boxes = word_boxes_east
            
            if not word_boxes:
                print("No words detected, processing entire image...")
                x = preprocess_image(image_path)
                y_pred = model.predict(x, verbose=0)
                texts = ctc_decode(y_pred)
                return texts[0] if texts else ""
            
            print(f"Found {len(word_boxes)} word(s)")
            
            # Group words into lines for multi-line text
            lines = group_words_into_lines(word_boxes)
            print(f"Grouped into {len(lines)} line(s)")
            
            # OCR each line
            recognized_lines = []
            for line_idx, line_boxes in enumerate(lines):
                recognized_words = []
                for _, (startX, startY, endX, endY) in enumerate(line_boxes):
                    # Extract word region with padding
                    padding = 5
                    startY = max(0, startY - padding)
                    endY = min(image.shape[0], endY + padding)
                    startX = max(0, startX - padding)
                    endX = min(image.shape[1], endX + padding)
                    
                    word_img = image[startY:endY, startX:endX]
                    
                    if word_img.size == 0:
                        continue
                    
                    # Preprocess and recognize
                    x = preprocess_word_image(word_img)
                    y_pred = model.predict(x, verbose=0)
                    texts = ctc_decode(y_pred)
                    word_text = texts[0] if texts else ""
                    
                    if word_text:
                        recognized_words.append(word_text)
                
                if recognized_words:
                    line_text = ' '.join(recognized_words)
                    recognized_lines.append(line_text)
                    print(f"  Line {line_idx + 1}: '{line_text}'")
            
            # Join lines with newlines
            return '\n'.join(recognized_lines)
            
        except Exception as e:
            print(f"Word detection failed: {e}")
            print("Falling back to whole image OCR...")
            import traceback
            traceback.print_exc()
            x = preprocess_image(image_path)
            y_pred = model.predict(x, verbose=0)
            texts = ctc_decode(y_pred)
            return texts[0] if texts else ""
    else:
        print(f"Processing image: {image_path}")
        x = preprocess_image(image_path)
        
        print("Running OCR...")
        y_pred = model.predict(x, verbose=0)
        
        texts = ctc_decode(y_pred)
        return texts[0] if texts else ""


def visualize_word_boxes(image_path, word_boxes, output_path=None):
    """Draw bounding boxes around detected words and save/display the result"""
    # Load original image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image for visualization: {image_path}")
        return
    
    # Create a copy for drawing
    vis_image = image.copy()
    
    # Draw each word box
    for i, (startX, startY, endX, endY) in enumerate(word_boxes):
        # Draw rectangle
        cv2.rectangle(vis_image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        # Add box number
        cv2.putText(vis_image, f"W{i+1}", (startX, startY-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save or generate output path
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_boxes{ext}"
    
    cv2.imwrite(output_path, vis_image)
    print(f"Visualization saved to: {output_path}")
    
    return output_path


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_single_image.py <image_path> [--detect-words] [--visualize] [model_path]")
        print("Example: python test_single_image.py test_image.png")
        print("Example: python test_single_image.py test_image.png --detect-words")
        print("Example: python test_single_image.py test_image.png --detect-words --visualize")
        sys.exit(1)
    
    image_path = sys.argv[1]
    detect_words = '--detect-words' in sys.argv
    visualize = '--visualize' in sys.argv
    
    # Get model path (skip flags)
    model_args = [arg for arg in sys.argv[2:] if not arg.startswith('--')]
    model_path = model_args[0] if model_args else 'models/crnn_ocr_ctc_1m_checkpoint.h5'
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    try:
        # If visualize is requested with word detection, get the boxes
        if visualize and detect_words:
            print("Detecting word boxes for visualization...")
            # Use spacing-based detection (same as recognition)
            _, word_boxes = detect_words_spacing(image_path)
            
            # Fallback to EAST if needed
            if not word_boxes or len(word_boxes) <= 1:
                print("Spacing found few words, trying EAST detection...")
                _, word_boxes = detect_words_east(image_path)
            
            if word_boxes:
                print(f"Visualizing {len(word_boxes)} word box(es)")
                visualize_word_boxes(image_path, word_boxes)
            else:
                print("No word boxes detected for visualization")
        
        recognized_text = recognize_text(image_path, model_path, detect_words=detect_words)
        print(f"\n{'='*50}")
        print(f"Recognized text: {recognized_text}")
        print(f"{'='*50}\n")
    except Exception as e:
        print(f"Error during OCR: {e}")
        import traceback
        traceback.print_exc()
