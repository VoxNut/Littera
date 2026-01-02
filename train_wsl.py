"""
Training script for CRNN OCR model - Optimized for WSL with GPU
Run this in WSL: python3 train_wsl.py
"""
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
import PIL
from PIL import ImageFile
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for WSL
import matplotlib.pyplot as plt
# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU detected: {len(gpus)} device(s)")
        print(f"  {gpus[0].name}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("⚠ No GPU detected - training will use CPU (slower)")

# Configuration
print("\n=== Configuration ===")
IMG_HEIGHT, IMG_WIDTH = 32, 128
BATCH_SIZE = 16  # Reduced from 64 to 32 to fit in GPU memory
TIMESTEPS = 31
EPOCHS = 10
NUM_SAMPLES = None

# Setup paths - WSL can access Windows drives via /mnt/
BASE_DIR = Path("/mnt/c/Users/ASUS/Littera")
MODEL_SAVE_DIR = BASE_DIR / "models"
MODEL_SAVE_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path("/mnt/c/Users/ASUS/huggingface_cache")
CACHE_DIR.mkdir(exist_ok=True)

from datasets import load_dataset


print(f"Base directory: {BASE_DIR}")
print(f"Model save directory: {MODEL_SAVE_DIR}")
print(f"Cache directory: {CACHE_DIR}")

# Set environment for HuggingFace cache
os.environ['HF_HOME'] = str(CACHE_DIR)
os.environ['HF_DATASETS_CACHE'] = str(CACHE_DIR / 'datasets')

# Allow truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Character set - includes uppercase, lowercase, and digits
charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
char_to_id = {c: i for i, c in enumerate(charset)}
num_classes = len(charset)

print(f"Character set size: {num_classes}")
print(f"Charset: {charset}")

# Encoding function
def encode_text(t: str):
    """Encode text to label IDs"""
    return [char_to_id[c] for c in t if c in char_to_id]

# Data generator
def gen(examples):
    """Generator for creating batches from dataset with augmentation"""
    import random
    
    for ex in examples:
        try:
            img = ex["image"].convert("L").resize((IMG_WIDTH, IMG_HEIGHT), PIL.Image.BILINEAR)
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            # Add data augmentation (random variations)
            if random.random() > 0.3:  # 70% of the time apply augmentation
                # Random brightness adjustment
                if random.random() > 0.5:
                    brightness_factor = random.uniform(0.8, 1.2)
                    img_array = np.clip(img_array * brightness_factor, 0, 1)
                
                # Random noise
                if random.random() > 0.7:
                    noise = np.random.normal(0, 0.02, img_array.shape)
                    img_array = np.clip(img_array + noise, 0, 1)
            
            img_array = img_array[..., None]  # (H, W, 1)
            label_ids = np.array(encode_text(ex["label"]), dtype=np.int32)
            
            if len(label_ids) == 0:
                continue
                
            yield img_array, label_ids, np.int32(len(label_ids))
        except (OSError, PIL.UnidentifiedImageError, Exception):
            continue

# Load dataset
print("\n=== Loading Dataset ===")
if NUM_SAMPLES is None:
    print("Loading FULL MJSynth dataset...")
    ds = load_dataset("priyank-m/MJSynth_text_recognition", split="train")
else:
    print(f"Loading {NUM_SAMPLES:,} samples from MJSynth dataset...")
    ds = load_dataset("priyank-m/MJSynth_text_recognition", split=f"train[:{NUM_SAMPLES}]")
print(f"✓ Loaded {len(ds):,} samples successfully!")

# Split train/validation
print("\n=== Splitting Dataset ===")
split = ds.train_test_split(test_size=0.1, seed=42)
train_raw, val_raw = split["train"], split["test"]
print(f"Train samples: {len(train_raw):,}")
print(f"Validation samples: {len(val_raw):,}")

# Create TensorFlow datasets
def make_tfds(hf_dataset):
    """Convert HuggingFace dataset to TensorFlow dataset"""
    output_signature = (
        tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    ds_tf = tf.data.Dataset.from_generator(
        lambda: gen(hf_dataset), 
        output_signature=output_signature
    )
    ds_tf = ds_tf.padded_batch(
        BATCH_SIZE,
        padded_shapes=(
            (IMG_HEIGHT, IMG_WIDTH, 1),
            (None,),
            (),
        ),
        padding_values=(0.0, -1, 0),
        drop_remainder=True
    ).prefetch(tf.data.AUTOTUNE)
    return ds_tf

print("\n=== Creating TensorFlow Datasets ===")
train_ds = make_tfds(train_raw)
val_ds = make_tfds(val_raw)
print("✓ Datasets ready!")

# Build CRNN model
def build_crnn(num_classes):
    """Build CRNN architecture for OCR - Enhanced version"""
    inp = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="image")

    # Deeper CNN with more filters
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(inp)
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.2)(x)
    
    x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.2)(x)
    
    x = keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D((2, 1))(x)
    x = keras.layers.Dropout(0.2)(x)
    
    x = keras.layers.Conv2D(512, 3, padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 1))(x)
    x = keras.layers.Conv2D(512, 2, padding="valid", activation="relu")(x)

    x = keras.layers.Reshape(target_shape=(31, 512))(x)
    
    # Deeper LSTM
    x = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True, dropout=0.2))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True, dropout=0.2))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.2))(x)
    
    out = keras.layers.Dense(num_classes + 1, activation="softmax")(x)
    return keras.Model(inp, out, name="crnn")

print("\n=== Building Model ===")

# Build fresh model (don't load from checkpoint)
checkpoint_path = MODEL_SAVE_DIR / "crnn_ocr_ctc_full_checkpoint.h5"

print("  Building new model from scratch...")
base_model = build_crnn(num_classes)

print(f"✓ Base model ready")
print(f"  Input shape: {base_model.input.shape}")
print(f"  Output shape: {base_model.output.shape}")

# Build training model with CTC loss
labels = keras.Input(shape=(None,), dtype=tf.int32, name="labels")
input_length = keras.Input(shape=(1,), dtype=tf.int32, name="input_length")
label_length = keras.Input(shape=(1,), dtype=tf.int32, name="label_length")

logits = base_model.output

def ctc_loss_layer(args):
    """CTC loss function"""
    y_true, y_pred, in_len, lab_len = args
    return keras.backend.ctc_batch_cost(y_true, y_pred, in_len, lab_len)

loss_out = keras.layers.Lambda(ctc_loss_layer, name="ctc_loss")(
    [labels, logits, input_length, label_length]
)

train_model = keras.Model(
    inputs=[base_model.input, labels, input_length, label_length],
    outputs=loss_out,
)

train_model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=lambda y_true, y_pred: y_pred
)

print("✓ Training model built with CTC loss")
print("  Using learning rate: 1e-3")

# Pack batches
def pack_batch(images, labels_batch, label_lens):
    """Pack batch data for training"""
    bsz = tf.shape(images)[0]
    in_len = tf.fill([bsz, 1], TIMESTEPS)
    lab_len = tf.expand_dims(label_lens, axis=1)
    inputs = {
        "image": images,
        "labels": labels_batch,
        "input_length": in_len,
        "label_length": lab_len,
    }
    y = tf.zeros((bsz, 1), dtype=tf.float32)
    return inputs, y

train_data = train_ds.map(pack_batch)
val_data = val_ds.map(pack_batch)

# Setup callbacks
final_model_path = MODEL_SAVE_DIR / "crnn_ocr_ctc_full.h5"

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,  
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,  # Reduced from 3 to 2 - adjust LR faster
        min_lr=1e-7,  # Allow even smaller learning rate
        verbose=1
    )
]

# Start training
print("\n" + "="*60) 
print("STARTING TRAINING FROM SCRATCH")
print("="*60)
print(f"Total epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Training samples: {len(train_raw):,}")
print(f"Validation samples: {len(val_raw):,}")
print(f"Checkpoint: {checkpoint_path}")
print("="*60 + "\n")

history = train_model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Save final model
print(f"\n✓ Saving final model to: {final_model_path}")
base_model.save(str(final_model_path))

# Save training history plot
print("✓ Saving training history plot...")
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Training History')
plt.legend()
plt.grid(True)
plt.savefig(str(MODEL_SAVE_DIR / 'training_history.png'))
print(f"  Plot saved to: {MODEL_SAVE_DIR / 'training_history.png'}")

# Print summary
print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"Best validation loss: {min(history.history['val_loss']):.4f}")
print(f"Final model: {final_model_path}")
print(f"Model size: {final_model_path.stat().st_size / (1024**2):.2f} MB")
print(f"Checkpoint: {checkpoint_path}")
print("="*60)
