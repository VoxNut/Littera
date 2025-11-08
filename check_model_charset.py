"""
Kiểm tra charset của model đã train
"""
from pathlib import Path
from tensorflow import keras

# Đường dẫn model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'models' / 'crnn_ocr_ctc_1m_checkpoint.h5'

print("="*70)
print("KIỂM TRA CHARSET CỦA MODEL")
print("="*70)

# Load model (bỏ qua custom objects để tránh lỗi ctc_loss_layer)
print(f"\nLoading model from: {MODEL_PATH}")
try:
    # Thử load model hoàn chỉnh
    model = keras.models.load_model(MODEL_PATH, compile=False)
except (ValueError, TypeError) as e:
    print(f"Warning: Cannot load full model ({e})")
    print("Loading base model only (without training wrapper)...")
    
    # Load với custom_objects để bỏ qua Lambda layer
    def ctc_loss_layer(args):
        return args[1]  # dummy function
    
    model = keras.models.load_model(
        MODEL_PATH, 
        custom_objects={'ctc_loss_layer': ctc_loss_layer},
        compile=False
    )
    
    # Nếu vẫn lỗi, lấy base model từ inputs
    if hasattr(model, 'layers'):
        for layer in model.layers:
            if hasattr(layer, 'output_shape') and len(layer.output_shape) == 3:
                if layer.output_shape[-1] > 50:  # Tìm layer output cuối (Dense layer)
                    model = keras.Model(inputs=model.input, outputs=layer.output)
                    print(f"Extracted base model from layer: {layer.name}")
                    break

# Lấy output shape
output_shape = model.output_shape
print(f"\nModel output shape: {output_shape}")
print(f"  - Batch size: {output_shape[0]}")
print(f"  - Timesteps: {output_shape[1]}")
print(f"  - Classes (bao gồm blank): {output_shape[2]}")

num_classes_with_blank = output_shape[2]
num_classes = num_classes_with_blank - 1  # Trừ blank class

print(f"\nNumber of character classes (không bao gồm blank): {num_classes}")

# So sánh với các charset phổ biến
charsets = {
    "Chữ thường + số": "abcdefghijklmnopqrstuvwxyz0123456789",  # 36 ký tự
    "Chữ HOA + thường + số": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",  # 62 ký tự
}

print("\n" + "="*70)
print("SO SÁNH VỚI CHARSET PHỔ BIẾN:")
print("="*70)

for name, charset in charsets.items():
    match = "✓ KHỚP!" if len(charset) == num_classes else "✗"
    print(f"{match} {name}: {len(charset)} ký tự")
    if len(charset) == num_classes:
        print(f"      Charset: {charset}")

print("\n" + "="*70)
print("KẾT LUẬN:")
print("="*70)

if num_classes == 36:
    print("Model được train với CHARSET CHỈ CHỮ THƯỜNG + SỐ")
    print("  → Không thể output chữ HOA")
    print("  → Cần train lại với charset 62 ký tự")
elif num_classes == 62:
    print("Model được train với CHARSET CHỮ HOA + THƯỜNG + SỐ")
    print("  → Có thể output cả chữ hoa và chữ thường")
    print("  → Nếu vẫn output toàn chữ HOA, đó là vấn đề training (bias/data)")
else:
    print(f"Model có {num_classes} ký tự (không khớp charset chuẩn)")
    print("  → Custom charset hoặc model đặc biệt")

print("="*70)
