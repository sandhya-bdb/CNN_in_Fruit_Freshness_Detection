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
  - Replace the final FC layer with `Dropout(0.2)` + `Linear(in_features, num_classes)`  
- **Classes**: `Fresh`, `Spoiled`  
- **Training data**: ~16,000 images  
- **Target categories**: 16 fruit-types (e.g., Banana, Lemon, Lulo, Mango, Orange, Strawberry, Tamarillo, Tomato…)  
- **Performance**: > 90 % accuracy on validation set  

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
   git clone https://github.com/your-username/fruit-freshness-detection.git
   cd fruit-freshness-detection
## ⚙️ Install dependencies
pip install -r requirements.txt
## Ensure model weights are placed
 Save your trained weights file as: model/saved_model.pth
## Run the Streamlit app
 streamlit run app.py
## Use the app
Drag or upload a fruit image (supported formats: JPG, PNG)

The interface will show whether the fruit is Fresh or Spoiled
# 📂 Project Structure
```bash
fruit-freshness-detection/
│
├── app.py # Streamlit front-end
├── model_helper.py # Model definition & prediction logic
├── model/
│ └── saved_model.pth # Trained model weights
├── data/ # (Optional) raw/train/val/test splits
├── requirements.txt # Python dependencies
└── README.md # Project overview
```
# 📊 Training & Validation Overview
Built a CNN from scratch (before using transfer learning) for experimentation.

Applied data augmentation: random flip, rotation, color jitter, resizing, etc.

Split data into training, validation, and test sets to monitor generalisation.

Visualised representative images to understand class distribution and data quality.

Iterated over different numbers of epochs to optimise performance and avoid overfitting.
# 🛠️ Next Steps & Enhancements
Integrate with actual conveyor-belt camera feeds for live inference.

Extend the model to classify multiple fruits in a single image / crate.

Deploy on edge-devices (e.g., Jetson Nano, Raspberry Pi) for real-time on-site use.

Explore further model regularisation (batch-norm, dropout) and advanced architectures for improved accuracy.

