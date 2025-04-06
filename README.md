
# Anemia Prediction Model

![Anemia Prediction](https://img.shields.io/badge/Accuracy-98.95%25-brightgreen) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)

Welcome to the **Anemia Prediction Model**—a cutting-edge tool leveraging deep learning to predict anemia with near-perfect accuracy. Trained on a robust dataset and powered by PyTorch, this model achieves a remarkable **98.95% accuracy**, making it a reliable companion for medical diagnostics. Curious about how AI can transform healthcare? Dive in!

## 🌟 Features
- **High Accuracy**: Achieves 98.95% on a balanced test set (285 samples).
- **Flexible Input**: Predicts anemia from single samples or batches of data.
- **Real-World Ready**: Uses 7 key features: Gender, Hemoglobin, MCH, MCHC, MCV, Age, and Iron Levels.
- **Easy Deployment**: Load and predict with a single script in Google Colab or any Python environment.

## 📊 How It Works
This model is a PyTorch neural network trained on a dataset of 1421 samples (801 non-anemic, 620 anemic), balanced with SMOTE for optimal performance. It uses a 3-layer architecture (64-32-1) to classify individuals as "Anemic" or "Not Anemic" based on blood metrics and demographic data. The result? A tool that’s both precise and practical.

### Sample Prediction
Input: `[1, 11.5, 22.0, 30.0, 85.0, 30, 10.0]`  
Output: `"Anemic"`

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Libraries: `pandas`, `numpy`, `torch`, `joblib`
- Google Drive access (model files stored there)

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/anemia-prediction.git
   cd anemia-prediction
