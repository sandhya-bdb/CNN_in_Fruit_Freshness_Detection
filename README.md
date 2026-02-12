# 🍓 Fruit Freshness Detection using CNN with ResNet50 

A deep learning-powered system that classifies fruit images as **Fresh** or **Spoiled**.  
This application allows you to drag & drop an image of a fruit and get an instant freshness prediction — designed for real-world use in warehouse conveyor-belt setups.

---

## 🚀 Project Overview  
In modern fruit-processing warehouses, quality control is critical. This project aims to automate the inspection process using high-speed cameras and convolutional neural networks (CNNs).  
- Images of fruit crates are captured in real time as they move on conveyor belts.  
- A CNN model classifies each fruit image as “Fresh” or “Spoiled”.  
- The system streamlines inspection, reduces human error, and helps minimise wastage.

---

## 🧠 Model Details  
- **Base architecture**: ResNet50 (pre-trained on ImageNet)  
- **Customisation**:  
  - Freeze all layers except `layer4` + the final fully connected layer  
  - Replace the final FC layer with `Dropout(0.4)` + `Linear(in_features, num_classes)`  
- **Classes**: `Fresh`, `Spoiled`  
- **Training data**: ~16,000 images  
- **Target categories**: 16 fruit-types (e.g., Banana, Lemon, Lulo, Mango, Orange, Strawberry, Tamarillo, Tomato…)  
- **Performance**: tuned runs can reach ~99% validation accuracy  

---

## 🧰 Technology Stack  
- Python 3.x  
- PyTorch (for model definition & training)  
- torchvision (for data loaders & transforms)  
- PIL (for image handling)  
- Streamlit (for front-end drag & drop app)  


---

## ⚙️ Setup Instructions  

1. **Clone this repository**  
   ```bash
   git clone https://github.com/sandhya-bdb/CNN_in_Fruit_Freshness_Detection.git
   cd CNN_in_Fruit_Freshness_Detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model weights are placed**
   - Save trained weights as `Streamlit/model/saved_model.pth`

4. **Run the Streamlit app**
   ```bash
   cd Streamlit
   streamlit run app.py
   ```

5. **Use the app**
   - Upload a fruit image (`.jpg`, `.jpeg`, `.png`)
   - The app returns the predicted class (`Fresh` or `Spoiled`) and confidence score

## 🧪 Reproducible Tuned Training
Use the new script-based pipeline to reproduce and improve your tuned setup (instead of relying only on notebook state).

1. **Train with tuned defaults (`lr=1e-5`, `dropout=0.4`)**
   ```bash
   python training/train_resnet.py \
     --dataset-dir /path/to/FRUIT-16K \
     --output-dir artifacts/tuned_run \
     --epochs 10 \
     --batch-size 32 \
     --lr 1e-5 \
     --dropout 0.4 \
     --weight-decay 1e-4
   ```

2. **Train with stricter realistic evaluation (recommended when 99% looks inflated)**
   ```bash
   python training/train_resnet.py \
     --dataset-dir /path/to/FRUIT-16K \
     --output-dir artifacts/realistic_eval \
     --preset realistic_eval \
     --epochs 8
   ```

3. **Run random-search tuning**
   ```bash
   python training/tune_resnet.py \
     --dataset-dir /path/to/FRUIT-16K \
     --output-dir tuning_runs \
     --trials 8 \
     --epochs 8 \
     --preset realistic_eval
   ```

4. **Training outputs**
   - `best_model.pth`: best checkpoint by validation accuracy
   - `metrics.json`: epoch history + val/test accuracy
   - `classes.json`: class index order
   - `tuning_summary.json`: sorted trial leaderboard (for tuner runs)

# 📂 Project Structure
```bash
CNN_in_Fruit_Freshness_Detection/
│
├── Streamlit/
│   ├── app.py                    # Streamlit front-end
│   ├── model_helper.py           # Model definition & prediction logic
│   └── model/saved_model.pth     # Trained model weights
├── notebook/model.ipynb          # Training workflow notebook
├── training/
│   ├── train_resnet.py           # Reproducible tuned training pipeline
│   └── tune_resnet.py            # Random-search hyperparameter tuning
├── requirements.txt              # Python dependencies
└── README.md                     # Project overview
```
# 📊 Training & Validation Overview
- Built baseline CNN models before adopting transfer learning with ResNet50.

- Applied data augmentation (random flip, rotation, color jitter, resize) to improve generalisation.

- Partitioned data into training, validation, and test sets.

- Visualised representative samples to understand class balance and data quality.

- Tuned hyperparameters (learning rate, epochs, dropout) to avoid overfitting and maximise validation performance.
# 🛠️ Next Steps & Enhancements
Integrate with actual conveyor-belt camera feeds for live inference.

Extend the model to classify multiple fruits in a single image / crate.

Deploy on edge-devices (e.g., Jetson Nano, Raspberry Pi) for real-time on-site use.

Explore further model regularisation (batch-norm, dropout) and advanced architectures for improved accuracy.
