"""
data_loader.py
Loads the Adult Income dataset from OpenML.
Target: whether income >50K
Sensitive attributes: sex, race
"""
from sklearn.datasets import fetch_openml
import pandas as pd

def load_adult_data():
    """
    Downloads the Adult Income dataset and returns a cleaned DataFrame.
    """
    print("Downloading Adult Income dataset from OpenML (first run may take a minute)...")
    data = fetch_openml(data_id=1590, as_frame=True, parser="auto") # 1590 = Adult dataset on OpenML
    df = data.frame.copy()
    
    # Target column in this version is 'class': '>50K' or '<=50K'
    y = (df["class"] == ">50K").astype(int)
    
    # Keep sensitive attributes separate for fairness measurement
    sensitive = df[["sex", "race"]].copy()
    
    # Drop target and sensitive attrs from feature set used for training
    X = df.drop(columns=["class"])
    
    print(f"Loaded {len(df)} rows, {X.shape[1]} feature columns.")
    print(f"Income >50K rate overall: {y.mean():.2%}")
    return X, y, sensitive

if __name__ == "__main__":
    X, y, sensitive = load_adult_data()
    print("\nSample rows:")
    print(X.head())
    print("\nSensitive attribute distribution:")
    print(sensitive["sex"].value_counts())
    print(sensitive["race"].value_counts())