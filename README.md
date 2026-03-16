<div align="center">
  <h1>🍎 Fruit Freshness Detection (CNN + ResNet50) 🍌</h1>
  <p><i>A Deep Learning project for classifying fruit images as <b>Fresh</b> or <b>Spoiled</b> using Transfer Learning.</i></p>

  [![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)
</div>

<br>

<div align="center">
  <img src="https://images.unsplash.com/photo-1610832958506-aa56368176cf?q=80&w=2070&auto=format&fit=crop" alt="Fruit Freshness Banner" width="800">
</div>

---

## 📖 Overview

The **Fruit Freshness Detection** project aims to tackle the widespread problem of food waste and quality control in agriculture and retail. By leveraging deep learning specifically Convolutional Neural Networks (CNNs) and the **ResNet50** architecture we can accurately classify an image of a fruit as either **Fresh** or **Spoiled**.

This repository contains everything from data preprocessing, model training scripts, a Jupyter notebook for exploratory data analysis, to a fully functional **Streamlit web application** for real-time inference.

### ✨ Key Features
- **Transfer Learning (ResNet50)**: Fine-tuned an ImageNet-pretrained ResNet50 model for the binary classification task.
- **Data Augmentation Strategies**: Employs robust data augmentation (random resized crops, color jittering, rotation) to prevent overfitting.
- **Interactive Web App**: A beautiful Streamlit interface that allows users to upload images and instantly get predictions.
- **Flexible Training Scripts**: Includes a minimal script-based trainer (`train_resnet.py`) with configurable presets (e.g., `tuned` vs. `realistic_eval`).

---

## 📂 Project Structure

```text
CNN_in_Fruit_Freshness_Detection/
├── Streamlit/
│   ├── app.py                     # Main Streamlit application
│   ├── model_helper.py            # Model loading & inference logic
│   └── model/
│       └── saved_model.pth        # Trained PyTorch model weights (requires download/training)
├── notebook/
│   └── model.ipynb                # Jupyter notebook with EDA & training experiments
├── training/
│   └── train_resnet.py            # CLI script for model training
├── requirements.txt               # Main dependency file
├── requirement.txt                # Legacy dependency file (backward compatibility)
└── README.md                      # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have **Python 3.8+** installed. You will also need `git` to clone the repository.

### 2. Local Setup
Clone the repository and install the required dependencies:

```bash
git clone https://github.com/sandhya-bdb/CNN_in_Fruit_Freshness_Detection.git
cd CNN_in_Fruit_Freshness_Detection

# It is recommended to use a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

> **Note:** The `requirements.txt` file installs `streamlit`, `Pillow`, `torch`, `torchvision`, and `numpy`.

### 3. Running the Streamlit App
If you have the trained weights saved in `Streamlit/model/saved_model.pth`, you can launch the app directly:

```bash
cd Streamlit
streamlit run app.py
```
The application will launch in your default web browser (usually at `http://localhost:8501`). Simply upload a `.jpg`, `.jpeg`, or `.png` image of a fruit, and the model will predict its freshness!

---

## 🧠 Model Training

We provide a minimal yet powerful training script `training/train_resnet.py` for those who want to train the model from scratch or on a new dataset.

### Minimal Training Example
Run the script from the root repository directory:

```bash
python3 training/train_resnet.py \
  --dataset-dir "/absolute/path/to/FRUIT-16K" \
  --output-dir "artifacts/minimal_run" \
  --epochs 8 \
  --preset realistic_eval
```

### Script Presets
To make training configurations easier, we provide two built-in presets via the `--preset` argument:
- `tuned`: Focuses on raw mathematical accuracy, utilizing standard augmentation. Recommended if your test set closely aligns with the training domain.
- `realistic_eval`: Implements a stricter valid/test split, stronger regularization (dropout, higher weight decay), and aggressive data augmentation (`ColorJitter`, `RandomResizedCrop`). Use this to reduce overly optimistic metrics and ensure the model generalizes effectively.

### Training Artifacts
Upon successful training, the script outputs the following files in your specified `--output-dir`:
- `best_model.pth`: The PyTorch model weights achieving the highest validation accuracy.
- `metrics.json`: A detailed dictionary logging test accuracies, losses, epoch progress, and configuration parameters.

---

## ☁️ Colab Quickstart (GPU Training)
Training a ResNet50 model can be computationally heavy. Google Colab provides a free environment with GPU access, accelerating the training process significantly.

1. Open a new Google Colab Noteook.
2. Enable GPU: Navigate to **Runtime -> Change runtime type** and select **T4 GPU**.
3. Mount your Google Drive to access your datasets:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Clone and setup the repository:
   ```bash
   %cd /content
   !git clone https://github.com/sandhya-bdb/CNN_in_Fruit_Freshness_Detection.git
   %cd /content/CNN_in_Fruit_Freshness_Detection
   !pip install -r requirements.txt
   ```
5. Initiate training:
   ```bash
   !python training/train_resnet.py \
     --dataset-dir "/content/drive/MyDrive/datasets/FRUIT-16K" \
     --output-dir "/content/drive/MyDrive/CNN_outputs/minimal_run" \
     --preset realistic_eval \
     --epochs 10
   ```
6. Check your final metrics:
   ```bash
   !cat /content/drive/MyDrive/CNN_outputs/minimal_run/metrics.json
   ```

---

## 🤝 Contributing
Contributions are always welcome! Whether it's adding a new fruit class, improving the model's accuracy, or enhancing the UI—feel free to open a Pull Request.
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License
Distributed under the MIT License. See `LICENSE` for more information.

<br>
<div align="center">
  <b>Built with ❤️ by Sandhya B.D. Borah</b>
</div>
