# Training CRNN Model in WSL with GPU

## Prerequisites
✓ WSL2 installed
✓ Python 3 with TensorFlow[and-cuda] installed in WSL
✓ GPU detected in WSL (verified)

## Quick Start

### 1. Activate your WSL environment
```bash
cd /mnt/d/My\ WorkSpace/Littera
source wsl_littera_env/bin/activate
```

### 2. Install required packages
```bash
pip install datasets pillow matplotlib
```

### 3. Run training
```bash
python3 train_wsl.py
```

## What the script does:
- ✓ Uses your RTX 3050 GPU automatically
- ✓ Loads 1,000,000 samples from MJSynth dataset
- ✓ Trains for 20 epochs with early stopping
- ✓ Saves checkpoints to `models/crnn_ocr_ctc_1m_checkpoint.h5`
- ✓ Saves final model to `models/crnn_ocr_ctc_1m.h5`
- ✓ Creates training history plot

## Monitor training progress:
The script will show:
- GPU detection status
- Dataset loading progress
- Epoch progress with loss values
- Checkpoint saves
- Validation loss improvements

## Expected training time:
- With RTX 3050 GPU: ~2-4 hours for 1M samples
- Without GPU (CPU): ~24-48 hours

## Files created:
```
models/
├── crnn_ocr_ctc_1m_checkpoint.h5  (best checkpoint)
├── crnn_ocr_ctc_1m.h5              (final model)
└── training_history.png             (loss plot)
```

## Resume training if interrupted:
If training stops, you can resume by running the same command again.
The script will load the checkpoint and continue.

## View results in Windows:
After training completes, you can use the model in Windows:
- The model files are saved to `D:\My WorkSpace\Littera\models\`
- Use `test_single_image.py` in Windows to test the model
- Use Django web interface to run OCR

## Troubleshooting:

### GPU not detected:
```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Out of memory:
Reduce BATCH_SIZE in train_wsl.py (line 26):
```python
BATCH_SIZE = 32  # Reduce from 64
```

### Cache issues:
Clear HuggingFace cache:
```bash
rm -rf /mnt/d/huggingface_cache/datasets
```
