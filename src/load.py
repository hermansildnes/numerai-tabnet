import torch
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

# Gets prediction using the pretrained model
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
    
    scaler = joblib.load("scaler.save")
    
    # Apply same preprocessing to live data
    live_features_scaled = scaler.transform(live_data[features])
    
    # Convert to tensor
    live_tensor = torch.tensor(live_features_scaled, dtype=torch.float32).to(device)
    
    # Make predictions
    with torch.no_grad():
        predictions, _ = model(live_tensor)  # TabNet returns (predictions, sparse_loss)    
        submission = pd.Series(predictions.cpu().numpy().flatten(), index=live_data.index).to_frame("prediction")
    
    # Save submission to csv as numerai expects
    submission.to_csv("submission.csv")
    print(f"Submission saved with {len(submission)} predictions")
    
if __name__ == "__main__":
    main()