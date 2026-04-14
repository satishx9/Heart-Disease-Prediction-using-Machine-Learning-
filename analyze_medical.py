import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_dataset():
    df = pd.read_csv("data/heart_data.csv")
    
    # Check correlations with target
    correlations = df.corr()["target"].sort_values(ascending=False)
    print("===== Correlations with Target =====")
    print(correlations)
    
    # Check rows similar to user input
    # User Input: age=52, gender=1, cp=3, trestbps=140, chol=260, fbs=1, restecg=1, thalach=140, exang=1, oldpeak=2.5, slope=1, ca=1, thal=3
    
    print("\n===== Target Distribution for key features =====")
    print("\nExang vs Target:")
    print(pd.crosstab(df['exang'], df['target']))
    
    print("\nCA vs Target:")
    print(pd.crosstab(df['ca'], df['target']))
    
    print("\nThal vs Target:")
    print(pd.crosstab(df['thal'], df['target']))
    
    print("\nCP vs Target:")
    print(pd.crosstab(df['cp'], df['target']))

if __name__ == "__main__":
    analyze_dataset()
