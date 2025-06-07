import pandas as pd

# Clips predicted values between 0 and 1 as numerai expects. (Most predictions are, but ensures no outliers)
# Should be fixed in the model, but this is a quick fix for now.
def main():
    # Load the submission dataframe
    df = pd.read_csv("submission.csv")
    
    # Clip prediction values to be between 0 and 1
    df['prediction'] = df['prediction'].clip(lower=0, upper=1)
    
    # Save the corrected submission (overwrite submission.csv)
    df.to_csv("submission.csv", index=False)
    print(f"Fixed submission saved with {len(df)} predictions")
    print(f"Prediction range: {df['prediction'].min():.4f} to {df['prediction'].max():.4f}")

if __name__ == "__main__":
    main()