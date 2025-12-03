# 📚 Littera – OCR Recognition System Using CRNN

**Littera** is an advanced OCR (Optical Character Recognition) system developed using the CRNN architecture. It combines the power of **Convolutional Neural Networks (CNN)** for visual feature extraction and **Recurrent Neural Networks (RNN)** for sequential character prediction.  
The system can recognize multilingual text, translate it automatically, and operate in real-time from both images and live camera feeds.

![Example from dataset](/imgs/resultExample.png)
---
## 🧑‍💻 Development Team

| No. | Full Name           | Student ID |
| :---: |:--------------------|:----------:|
| 01 | Võ Minh Nhựt        |  23130226  |
| 02 | Lê Đại Nhân         |  23130215  |
| 03 | Nguyễn Hoàng Kỳ Anh |  23130010  |
| 04 | Nguyễn Đình Hiếu    |  23130107  |


---

## 🔑 Key Features

### 📝 Text Recognition
- Recognizes **English text** across multiple fonts: uppercase, lowercase, stylized fonts  
- Supports **image input** and **real-time camera input**

### 🔁 Automatic Translation
- Translates recognized text **between English ↔ Vietnamese**
- Ideal for translating signs, posters, books, magazines, and documents

### ⚡ Real-Time Processing
- Low-latency OCR for real-time applications  
- Suitable for AR translation, mobile apps, smart glasses, etc.

---

## 📐 System Architecture

### CRNN Model (CNN + RNN + CTC)
- **CNN**: Extracts spatial image features  
- **RNN (Bi-LSTM/GRU)**: Predicts character sequences  
- **CTC Loss**: Handles flexible sequence alignment and variable-length output

---

## 🎯 Performance    Metrics

| Metric                  | Description                                   |
|-------------------------|-----------------------------------------------|
| Character Accuracy (%)  | Correctness of individual character output    |
| Word Accuracy (%)       | Correctness of full word recognition          |

---

## Case Statistics
![Example from dataset](/imgs/case_distribution.png)


---

---

## Training History
![Training History](/models/training_history.png)


---

## 🛠️ Technology Stack

### Tools
- CUDA (optional GPU acceleration)

---

## 🚀 Getting Started

### ✅ Prerequisites
- Python 3.8+
- pip / conda
- CUDA-enabled GPU (recommended)
- OpenCV
- PyTorch

### 📥 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/littera-ocr.git
```

```bash
cd Littera
pip install -r requirements.txt
```
---
## Dataset
System was using dataset: [MJSynth_text_recognition](https://huggingface.co/datasets/priyank-m/MJSynth_text_recognition).

```bash
#Load dataset
print("Loading 1,000,000 samples from MJSynth dataset...")
ds = load_dataset("priyank-m/MJSynth_text_recognition", split="train[:1000000]")
print(f"Loaded {len(ds)} samples successfully!")
```