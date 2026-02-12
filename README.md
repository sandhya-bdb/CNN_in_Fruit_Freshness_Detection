# Fruit Freshness Detection (CNN + ResNet50)

A deep-learning project for classifying fruit images as **Fresh** or **Spoiled**.

## Overview
- Model: ResNet50 transfer learning (ImageNet pretrained)
- Task: Binary freshness classification
- Interface: Streamlit app for image upload + prediction
- Training: notebook workflow + minimal script-based trainer

## Project Structure
```text
CNN_in_Fruit_Freshness_Detection/
├── Streamlit/
│   ├── app.py
│   ├── model_helper.py
│   └── model/saved_model.pth
├── notebook/model.ipynb
├── training/train_resnet.py
├── requirement.txt
├── requirements.txt
└── README.md
```

## Local Setup (Streamlit Inference)
```bash
git clone https://github.com/sandhya-bdb/CNN_in_Fruit_Freshness_Detection.git
cd CNN_in_Fruit_Freshness_Detection
python3 -m pip install -r requirements.txt
cd Streamlit
streamlit run app.py
```

## Minimal Training Script
Run from repository root:

```bash
python3 training/train_resnet.py \
  --dataset-dir "/absolute/path/to/FRUIT-16K" \
  --output-dir "artifacts/minimal_run" \
  --preset realistic_eval \
  --epochs 8
```

### Presets
- `tuned`: More accuracy-oriented defaults
- `realistic_eval`: Harder split + stronger regularization to reduce overly optimistic results

### Training Output
The script writes:
- `best_model.pth`
- `metrics.json`

in the chosen `--output-dir`.

## Google Colab Quick Run
1. Enable GPU: `Runtime` -> `Change runtime type` -> `T4 GPU`
2. Run cells:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%cd /content
!git clone https://github.com/sandhya-bdb/CNN_in_Fruit_Freshness_Detection.git
%cd /content/CNN_in_Fruit_Freshness_Detection
!pip install -r requirements.txt
```

```bash
!python training/train_resnet.py \
  --dataset-dir "/content/drive/MyDrive/datasets/FRUIT-16K" \
  --output-dir "/content/drive/MyDrive/CNN_outputs/minimal_run" \
  --preset realistic_eval \
  --epochs 8
```

Check metrics:

```bash
!cat /content/drive/MyDrive/CNN_outputs/minimal_run/metrics.json
```

## Notes
- `requirements.txt` is the primary dependency file.
- `requirement.txt` is kept for backward compatibility.
