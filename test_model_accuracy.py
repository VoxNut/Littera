"""
Test OCR model accuracy using MJSynth dataset
"""
import os
from pathlib import Path

CACHE_DIR = Path("D:/huggingface_cache")
CACHE_DIR.mkdir(exist_ok=True)
os.environ['HF_HOME'] = str(CACHE_DIR)
os.environ['HF_DATASETS_CACHE'] = str(CACHE_DIR / 'datasets')

# Now import datasets and other libraries AFTER setting environment variables
import time
import numpy as np 
from PIL import Image
from tensorflow import keras
from tensorflow.keras.models import Model
from datasets import load_dataset

K = keras.backend

# Configuration matching your model
IMG_HEIGHT, IMG_WIDTH = 32, 128
CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
ID_TO_CHAR = {i: c for i, c in enumerate(CHARSET)}

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'models' / 'crnn_ocr_ctc_1m_checkpoint.h5'


# ============================================================
# CRITICAL: Define custom CTC loss BEFORE loading model
# ============================================================
def ctc_loss_layer(args):
    """Custom CTC loss function"""
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def load_inference_model(model_path):
    """
    Load model and extract inference part (only image input)
    Training model has 4 inputs, but we only need 1 for prediction
    """
    print("\n[1/4] Loading model...")
    
    try:
        # Load full model with custom objects
        print("  Loading full training model...")
        full_model = keras.models.load_model(
            model_path,
            custom_objects={'ctc_loss_layer': ctc_loss_layer},
            compile=False
        )
        print(f"  ✓ Model loaded: {len(full_model.inputs)} inputs, {len(full_model.outputs)} outputs")
        
        # If model has only 1 input, it's already an inference model
        if len(full_model.inputs) == 1:
            print("  ✓ Model is already in inference mode")
            return full_model
        
        # Extract inference model (image input only)
        print("  Extracting inference model (image input only)...")
        
        # Get image input (first input)
        image_input = full_model.inputs[0]
        print(f"  Image input shape: {image_input.shape}")
        
        # Find prediction output layer (before CTC loss)
        prediction_output = None
        for layer in full_model.layers:
            # Look for Dense layer with correct output size
            if hasattr(layer, 'units') and layer.units == len(CHARSET) + 1:
                prediction_output = layer.output
                print(f"  Found prediction layer: {layer.name}")
                break
        
        if prediction_output is None:
            # Alternative: find layer with correct output shape
            for layer in reversed(full_model.layers):
                if len(layer.output_shape) == 3 and layer.output_shape[-1] == len(CHARSET) + 1:
                    prediction_output = layer.output
                    print(f"  Found prediction layer: {layer.name}")
                    break
        
        if prediction_output is None:
            print("  ⚠️  Could not find prediction layer, using first output")
            prediction_output = full_model.outputs[0]
        
        # Create inference model
        inference_model = Model(inputs=image_input, outputs=prediction_output)
        print(f"  ✓ Inference model created")
        print(f"    Input: {inference_model.input_shape}")
        print(f"    Output: {inference_model.output_shape}")
        
        return inference_model
        
    except Exception as e:
        print(f"  ❌ Error loading model: {e}")
        raise


def preprocess_image(pil_img):
    """Preprocess image same as in views.py"""
    img = pil_img.convert('L').resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
    x = np.array(img, dtype=np.float32) / 255.0
    x = x[..., None]
    x = np.expand_dims(x, 0)
    return x


def ctc_decode(y_pred):
    """Decode predictions same as in views.py"""
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


def calculate_metrics(predictions, ground_truths):
    """Calculate word and character accuracy"""
    correct_words = 0
    correct_words_case_insensitive = 0
    total_chars = 0
    correct_chars = 0
    correct_chars_case_insensitive = 0

    for pred, gt in zip(predictions, ground_truths):
        # Word-level accuracy (exact match with case)
        if pred == gt:
            correct_words += 1
        
        # Word-level accuracy (case-insensitive)
        if pred.lower() == gt.lower():
            correct_words_case_insensitive += 1

        # Character-level accuracy (with case)
        min_len = min(len(pred), len(gt))
        for i in range(min_len):
            total_chars += 1
            if pred[i] == gt[i]:
                correct_chars += 1
            if pred[i].lower() == gt[i].lower():
                correct_chars_case_insensitive += 1

        # Penalize length differences
        total_chars += abs(len(pred) - len(gt))

    word_accuracy = correct_words / len(predictions) * 100 if predictions else 0
    word_accuracy_ci = correct_words_case_insensitive / len(predictions) * 100 if predictions else 0
    char_accuracy = correct_chars / total_chars * 100 if total_chars > 0 else 0
    char_accuracy_ci = correct_chars_case_insensitive / total_chars * 100 if total_chars > 0 else 0

    return {
        'word_acc': word_accuracy,
        'word_acc_ci': word_accuracy_ci,
        'char_acc': char_accuracy,
        'char_acc_ci': char_accuracy_ci,
        'correct_count': correct_words,
        'correct_count_ci': correct_words_case_insensitive
    }


def test_on_dataset(num_samples=1000, split='test', start_idx=0):
    """
    Test model on MJSynth dataset

    Args:
        num_samples: Number of samples to test (default 1000)
        split: 'train', 'val', or 'test' (default 'test')
        start_idx: Starting index in the dataset (default 0)
    """
    print("="*70)
    print(f"Testing OCR Model on MJSynth {split.upper()} Dataset")
    print(f"Samples: {num_samples} (starting from index {start_idx})")
    print("="*70)

    # Load inference model (handles both 1-input and 4-input models)
    model = load_inference_model(MODEL_PATH)

    # Load dataset
    print(f"\n[2/4] Loading {split} dataset...")
    end_idx = start_idx + num_samples
    split_str = f"{split}[{start_idx}:{end_idx}]"
    ds = load_dataset("priyank-m/MJSynth_text_recognition", split=split_str)
    print(f"✓ Loaded {len(ds)} samples")

    # Run predictions
    print(f"\n[3/4] Running predictions...")
    predictions = []
    ground_truths = []
    errors = []

    start_time = time.time()

    for i, sample in enumerate(ds):
        try:
            # Preprocess
            x = preprocess_image(sample['image'])

            # Predict (now works with inference model)
            y = model.predict(x, verbose=0)
            pred_text = ctc_decode(y)[0]

            # Ground truth
            gt_text = sample['label']

            predictions.append(pred_text)
            ground_truths.append(gt_text)

            # Show progress every 100 samples
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                speed = (i + 1) / elapsed
                print(f"  Progress: {i+1}/{num_samples} ({speed:.1f} samples/sec)")

        except Exception as e:
            errors.append((i, str(e)))
            print(f"  ERROR at sample {i}: {e}")

    elapsed_time = time.time() - start_time

    # Calculate metrics
    print(f"\n[4/4] Calculating metrics...")
    metrics = calculate_metrics(predictions, ground_truths)

    # Print results
    print("\n" + "="*70)
    print(f"RESULTS - {split.upper()} SET")
    print("="*70)
    print(f"Total samples tested:    {len(predictions)}")
    print(f"Correct predictions (exact):     {metrics['correct_count']}")
    print(f"Correct predictions (case-insensitive): {metrics['correct_count_ci']}")
    print(f"Incorrect predictions:   {len(predictions) - metrics['correct_count']}")
    print(f"Errors during testing:   {len(errors)}")
    print(f"\nWord Accuracy (exact):           {metrics['word_acc']:.2f}%")
    print(f"Word Accuracy (case-insensitive): {metrics['word_acc_ci']:.2f}%")
    print(f"Character Accuracy (exact):      {metrics['char_acc']:.2f}%")
    print(f"Character Accuracy (case-insensitive): {metrics['char_acc_ci']:.2f}%")
    print(f"\nTime taken:              {elapsed_time:.2f} seconds")
    print(f"Speed:                   {len(predictions)/elapsed_time:.1f} samples/sec")
    print("="*70)

    # Show some examples
    print("\nSample Predictions (first 10):")
    print("-"*70)
    for i in range(min(10, len(predictions))):
        match = "✓" if predictions[i] == ground_truths[i] else "✗"
        print(f"{i+1}. {match} Expected: '{ground_truths[i]:20s}' | Predicted: '{predictions[i]}'")

    # Show some errors if any
    if len(predictions) > metrics['correct_count']:
        print("\nSample Errors (first 10):")
        print("-"*70)
        error_count = 0
        for i in range(len(predictions)):
            if predictions[i] != ground_truths[i]:
                print(f"{i+1}. ✗ Expected: '{ground_truths[i]:20s}' | Predicted: '{predictions[i]}'")
                error_count += 1
                if error_count >= 10:
                    break

    print()
    return metrics


if __name__ == '__main__':
    print("\n" + "="*70)
    print("OCR MODEL ACCURACY TESTING")
    print("="*70)
    print("📌 NOTE: Testing on TEST set for proper evaluation")
    print("="*70)
    
    # RECOMMENDED: Test on TEST set (unseen data)
    print("\n🎯 Testing on TEST set (proper evaluation):")
    test_on_dataset(num_samples=10_000, split='test', start_idx=0)
    
    # Optional: Test on VALIDATION set
    # print("\n\n📊 Testing on VALIDATION set:")
    # test_on_dataset(num_samples=1000, split='val', start_idx=0)
    
    # Optional: Test on TRAINING set (for comparison only)
    # print("\n\n📚 Testing on TRAINING set (for comparison):")
    # test_on_dataset(num_samples=1000, split='train', start_idx=0)
    
    # Optional: Larger test (takes longer)
    # print("\n\n🚀 Large scale test (10,000 samples):")
    # test_on_dataset(num_samples=10000, split='test', start_idx=0)