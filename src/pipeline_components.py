import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow.sklearn

# -----------------------------------------------------------
# COMPONENT 1: LOAD DATA
# -----------------------------------------------------------
def load_data(file_path):
    df = pd.read_csv(file_path)
    print(f"Loaded dataset with shape {df.shape}")
    return df


# -----------------------------------------------------------
# COMPONENT 2: PREPROCESSING
# -----------------------------------------------------------
def preprocess_data(df):
    # California housing CSV uses "TARGET"
    X = df.drop("TARGET", axis=1)
    y = df["TARGET"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# -----------------------------------------------------------
# COMPONENT 3: MODEL TRAINING
# -----------------------------------------------------------
def train_model(X_train, X_test, y_train, y_test):

    # --- DO NOT start a new run here; use the active run in pipeline.py ---
    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    # Log metrics to the active run
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "linear_regression_model")

    print(f"Model trained. MSE: {mse}")
    return model
