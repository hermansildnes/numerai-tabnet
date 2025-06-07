import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import json
from numerapi import NumerAPI
from TabNet import TabNet
from sklearn.preprocessing import StandardScaler
import joblib


def main():
    load_data() # Download data if not already present
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")  # Force CPU for debugging
    print(f"Using device: {device}")

    print("Loading data...")

    feature_metadata = json.load(open("features.json"))
    features = feature_metadata["feature_sets"]["medium"]
    data = pd.read_parquet("train.parquet", columns=["era", "target"]+features).dropna()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(data[features])
    print("Saving scaler...")
    joblib.dump(scaler, "scaler.save")
    


    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train = torch.tensor(data["target"].values, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=1024, shuffle=True, num_workers=4, pin_memory=True if device.type=="cuda" else False, persistent_workers=True if device.type=="cuda" else False)
    print("Data loaded successfully.")

    model = TabNet(inp_dim=len(features), final_out_dim=1, n_d=64, n_a=64, n_shared=2, n_ind=2, n_steps=5,relax=1.3,vbs=128).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()

    print("Starting training...")
    for epoch in range(125):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs, sparse_loss = model(X_batch)
            pred_loss = criterion(outputs, y_batch)
            total_loss = pred_loss + 1e-3 * sparse_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += total_loss.item()
        
        print(f"Epoch {epoch+1}/{125}, Train Loss: {train_loss/len(train_loader):.4f}")


def load_data():
    api = NumerAPI()
    api.download_dataset(
        "v5.0/train.parquet",
        "train.parquet"
        )
    api.download_dataset(
        f"v5.0/features.json",
        "features.json"
        )
    api.download_dataset(
	    "v5.0/live.parquet",
	    "live.parquet"
        )


if __name__ == "__main__":
    main()