from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load dataset
data = fetch_california_housing(as_frame=True)

# Convert to DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df['TARGET'] = data.target

# Save dataset
df.to_csv("data/raw/california_housing.csv", index=False)

print("Dataset saved to data/raw/california_housing.csv")
