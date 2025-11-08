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
from datasets import load_dataset

# Configuration matching your model
IMG_HEIGHT, IMG_WIDTH = 32, 128
# Model was trained with uppercase + lowercase + digits
CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
ID_TO_CHAR = {i: c for i, c in enumerate(CHARSET)}

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'models' / 'crnn_ocr_ctc_colab_500k.h5'


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


def test_on_dataset(num_samples=1000, start_idx=0):
    """
    Test model on MJSynth dataset

    Args:
        num_samples: Number of samples to test (default 1000)
        start_idx: Starting index in the dataset (default 0)
    """
    print("="*70)
    print(f"Testing OCR Model on MJSynth Dataset")
    print(f"Samples: {num_samples} (starting from index {start_idx})")
    print("="*70)

    # Load model
    print("\n[1/4] Loading model...")
    model = keras.models.load_model(MODEL_PATH, compile=False)
    print(f"✓ Model loaded from: {MODEL_PATH}")

    # Load dataset
    print(f"\n[2/4] Loading dataset split...")
    end_idx = start_idx + num_samples
    split_str = f"train[{start_idx}:{end_idx}]"
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

            # Predict
            y = model.predict(x, verbose=0)
            pred_text = ctc_decode(y)[0]

            # Ground truth - keep original case since model supports uppercase
            gt_text = sample['label']

            predictions.append(pred_text)
            ground_truths.append(gt_text)

            # Show progress every 100 samples
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                speed = (i + 1) / elapsed
                print(
                    f"  Progress: {i+1}/{num_samples} ({speed:.1f} samples/sec)")

        except Exception as e:
            errors.append((i, str(e)))
            print(f"  ERROR at sample {i}: {e}")

    elapsed_time = time.time() - start_time

    # Calculate metrics
    print(f"\n[4/4] Calculating metrics...")
    metrics = calculate_metrics(predictions, ground_truths)

    # Print results
    print("\n" + "="*70)
    print("RESULTS")
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
        print(
            f"{i+1}. {match} Expected: '{ground_truths[i]:20s}' | Predicted: '{predictions[i]}'")

    # Show some errors if any
    if len(predictions) > metrics['correct_count']:
        print("\nSample Errors (first 10):")
        print("-"*70)
        error_count = 0
        for i in range(len(predictions)):
            if predictions[i] != ground_truths[i]:
                print(
                    f"{i+1}. ✗ Expected: '{ground_truths[i]:20s}' | Predicted: '{predictions[i]}'")
                error_count += 1
                if error_count >= 10:
                    break

    print()


if __name__ == '__main__':
    # Test on different portions of the 500k training set

    # Option 1: Test on first 1000 samples
    print("\nTesting on FIRST 1000 samples of training data:")
    test_on_dataset(num_samples=1000, start_idx=0)

    # Option 2: Test on a middle portion (uncomment to run)
    # print("\n\nTesting on samples 250000-251000:")
    # test_on_dataset(num_samples=1000, start_idx=250000)

    # Option 3: Test on end portion (uncomment to run)
    # print("\n\nTesting on LAST 1000 samples (499000-500000):")
    # test_on_dataset(num_samples=1000, start_idx=499000)

    # Option 4: Test on larger sample (uncomment to run - will take longer)
    # print("\n\nTesting on 10000 samples:")
    # test_on_dataset(num_samples=10000, start_idx=0)
