
# anemia_predictor.py

# Import required libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from google.colab import drive
class AnemiaClassifier(nn.Module):
    def __init__(self, input_size):
        super(AnemiaClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def load_model_and_scaler():
    drive.mount('/content/drive', force_remount=True)
    drive_folder = '/content/drive/My Drive/Anemia_Model_PyTorch'
    model_path = f'{drive_folder}/anemia_pytorch_model.pkl'
    scaler_path = f'{drive_folder}/scaler_pytorch.pkl'

    loaded_model_data = joblib.load(model_path)
    input_size = loaded_model_data['input_size']  # 7 features
    model = AnemiaClassifier(input_size)
    model.load_state_dict(loaded_model_data['state_dict'])
    scaler = joblib.load(scaler_path)

    print("Model and scaler loaded successfully!")
    return model, scaler
def predict_anemia(model, scaler, new_data):
    if isinstance(new_data, (list, np.ndarray)):
        new_data = pd.DataFrame([new_data], columns=['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV', 'Age', 'Iron_Levels'])
    elif isinstance(new_data, pd.DataFrame):
        new_data = new_data[['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV', 'Age', 'Iron_Levels']]
    else:
        raise ValueError("Input must be a list, array, or DataFrame with 7 features")

    new_data_scaled = scaler.transform(new_data)
    new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        pred = model(new_data_tensor)
        results = (pred >= 0.5).float().numpy()
        return ["Anemic" if r == 1 else "Not Anemic" for r in results]
if __name__ == "__main__":
    !pip install pandas numpy torch joblib

    model, scaler = load_model_and_scaler()

    single_sample = [1, 11.5, 22.0, 30.0, 85.0, 30, 10.0]
    prediction = predict_anemia(model, scaler, single_sample)
    print("Prediction for single sample:", prediction[0])

    multiple_samples = pd.DataFrame({
        'Gender': [1, 0],
        'Hemoglobin': [11.5, 14.9],
        'MCH': [22.0, 25.4],
        'MCHC': [30.0, 28.3],
        'MCV': [85.0, 72.0],
        'Age': [30, 45],
        'Iron_Levels': [10.0, 20.0]
    })
    predictions = predict_anemia(model, scaler, multiple_samples)
    print("Predictions for multiple samples:")
    for i, pred in enumerate(predictions):
        print(f"Sample {i+1}: {pred}")
