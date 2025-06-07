import torch
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

def main():
    # Load the model
    model = torch.load("/home/error/downloads/TabNet_.pt", weights_only=False)
    model.eval()  # Set to evaluation mode
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load features metadata
    features_metadata = json.load(open("features.json"))
    features = features_metadata["feature_sets"]["medium"]
    
    # Load live data
    live_data = pd.read_parquet("live.parquet", columns=features)
    
    # CRITICAL: You need to apply the SAME preprocessing as during training!
    # You should have saved the scaler during training, but if not:
    # Option 1: Load the scaler if you saved it
    # scaler = joblib.load("scaler.pkl")  # You should save this during training
    
    # Option 2: If you don't have the saved scaler, you need to recreate it
    # This is NOT ideal as it might not match exactly what was used in training
    #print("WARNING: Recreating scaler - this should match your training preprocessing exactly!")
    
    # Load training data to fit scaler (this should match your training exactly) IF NEEDED.
    #train_data = pd.read_parquet("train.parquet", columns=["era", "target"] + features).dropna()
    scaler = joblib.load("scaler.save")
    #scaler = StandardScaler()
    #scaler.fit(train_data[features])  # Fit on training data
    
    # Apply same preprocessing to live data
    live_features_scaled = scaler.transform(live_data[features])
    
    # Convert to tensor
    live_tensor = torch.tensor(live_features_scaled, dtype=torch.float32).to(device)
    
    # Make predictions
    with torch.no_grad():
        predictions, _ = model(live_tensor)  # TabNet returns (predictions, sparse_loss)    
        submission = pd.Series(predictions.cpu().numpy().flatten(), index=live_data.index).to_frame("prediction")
    
    # Save submission - FIXED: use to_csv(), not save_csv()
    submission.to_csv("submission.csv")
    print(f"Submission saved with {len(submission)} predictions")
    
if __name__ == "__main__":
    main()