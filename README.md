# Glaucoma_Remastered
# 🧠 Glaucoma Detection using Deep Learning

A deep learning-based project for detecting **Glaucoma** from retinal fundus images using **TensorFlow/Keras** and **MobileNetV2**.

---

## 🚀 Overview

Glaucoma is a serious eye condition that can lead to vision loss if not detected early.
This project builds a model to classify retinal images into:

* **Glaucoma**
* **Normal**

using a lightweight and efficient CNN architecture.

---

## 🏗️ Tech Stack

* Python 🐍
* TensorFlow / Keras
* MobileNetV2 (Transfer Learning)
* OpenCV / PIL
* NumPy

---

## 📁 Project Structure

```
Glaucoma_Remastered/
│
├── train.py              # Model training script
├── predict.py            # Single image prediction
├── predict_cli.py        # CLI-based prediction
├── app.py                # (Optional) App interface
├── build_dataset.py      # Dataset preprocessing
├── glaucoma_model.h5     # Trained model (optional)
├── README.md
└── .gitignore
```

---

## 📊 Dataset

Due to size limitations, the dataset is not included in this repository.

---

## ⚙️ Setup & Installation

1. Clone the repository:

```
git clone https://github.com/yourusername/glaucoma-detection.git
cd glaucoma-detection
```

2. Create virtual environment:

```
python -m venv venv
venv\Scripts\activate   # Windows
```

3. Install dependencies:

```
pip install -r requirements.txt
```

---

## 🧪 Training the Model

```
python train.py
```

---

## 🔍 Prediction

### Single Image:

```
python predict.py test_image.png
```

### CLI Version:

```
python predict_cli.py test_image.png
```

---

## 📈 Model

* Base Model: **MobileNetV2**
* Transfer Learning applied
* Optimized for lightweight performance

---

## 🧠 Future Improvements

* Web interface (Flask / Streamlit)
* Better dataset balancing
* Model explainability (Grad-CAM)
* Deployment on cloud

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

---

## ⚠️ Disclaimer

This project is for educational purposes only and should not be used as a medical diagnostic tool.

---

## ✨ Author

**Tarun**
B.Tech CSE Student | AI Enthusiast | Artist

---

> “Not everything that sees… understands.
> This model tries to do both.”

