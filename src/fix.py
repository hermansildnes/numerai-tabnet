import pandas as pd

def main():
    # Load the submission dataframe
    df = pd.read_csv("~/downloads/submission.csv")
    
    # Clip prediction values to be between 0 and 1
    df['prediction'] = df['prediction'].clip(lower=0, upper=1)
    
    # Save the corrected submission
    df.to_csv("~/downloads/submission_fixed.csv", index=False)
    print(f"Fixed submission saved with {len(df)} predictions")
    print(f"Prediction range: {df['prediction'].min():.4f} to {df['prediction'].max():.4f}")

if __name__ == "__main__":
    main()