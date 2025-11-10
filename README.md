📚 Littera – OCR Recognition System Using CRNN

Littera is an advanced OCR (Optical Character Recognition) system developed using the CRNN architecture. It combines the power of Convolutional Neural Networks (CNN) for visual feature extraction and Recurrent Neural Networks (RNN/GRU/LSTM) for sequential character prediction.
The system can recognize multilingual text, translate it automatically, and operate in real-time from both images and live camera feeds.

🔑 Key Features
Text Recognition

Recognizes English text in multiple fonts: uppercase, lowercase, handwritten, stylized fonts

Supports image input and real-time camera input

Automatic Translation

Translates recognized text between English ↔ Vietnamese

Designed for instant translation of signs, posters, books, documents, and more

Real-Time Processing

Optimized for low-latency prediction

Suitable for mobile/embedded use cases, real-time AR translation, smart glasses, and live reading interfaces

📐 Design Architecture
CRNN Architecture (CNN + RNN + CTC)

CNN Layer: Extract visual features from image sequences

RNN Layer (Bi-LSTM/GRU): Model character dependencies for sequential output

CTC Loss: Enables flexible alignment and variable-length text prediction

Supporting Components

Preprocessing Pipeline: Noise removal, grayscale, normalization

Postprocessing Layer: Decoding, beam search, confidence scoring

Translation Engine: Uses transformer-based model for EN–VI translation

🎯 Performance Metrics

The system evaluates and tracks two main accuracy metrics:

Character Accuracy (%) – Measures correctness of each individual character

Word Accuracy (%) – Evaluates correctness at the word level

These metrics provide a comprehensive view of overall OCR performance.

🛠️ Technology Stack
Backend / Core

Python

PyTorch (CRNN implementation)

OpenCV (image processing)

TensorFlow (optional translation model integration)

Frontend

ElectronJS / PyQt (demo UI)

Additional Tools

CUDA for GPU acceleration (optional)

Tesseract (optional comparison baseline)

🚀 Getting Started
✅ Prerequisites

Python 3.8+

CUDA-enabled GPU (optional but recommended)

pip / conda

OpenCV

PyTorch

📥 Installation

Clone the repository:

git clone https://github.com/yourusername/littera-ocr.git


Navigate and install:

cd littera-ocr
pip install -r requirements.txt

▶️ Run the Demo
python run_demo.py

📸 Application Screenshots
Recognition From Image

OCR Image Input → Recognized Text

Real-Time Camera Recognition

Live OCR Feed

Translation Output

English → Vietnamese (or vice versa)

👨‍💻 Development Team
No.	Full Name   	    Student ID
01	Võ Minh Nhựt        23130226
02	Lê Đại Nhân 	    23130215
03	Nguyễn Hoàng Kỳ Anh	23130010
04	Nguyễn Đình Hiếu	23130107
📦 Project Modules
1. Preprocessing

Noise reduction

Grayscale conversion

Resize & normalize image

2. CRNN Model

CNN feature extractor

Bi-directional RNN

CTC decoder

3. Translation Engine

Neural translation model

Context-aware translation

4. Real-Time Pipeline

Capture camera frame

Run OCR + Translate

Display overlay text

📄 License

This project is licensed under the MIT License – See LICENSE file for details.