import mlflow
from pipeline_components import load_data, preprocess_data, train_model

def full_pipeline():
    # Set experiment
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("California-Housing-Pipeline")

    # Single top-level run
    with mlflow.start_run(run_name="Full-Pipeline-Run"):

        # Step 1: Load data
        df = load_data("data/raw/california_housing.csv")
        mlflow.log_param("rows", df.shape[0])
        mlflow.log_param("cols", df.shape[1])

        # Step 2: Preprocess
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
        mlflow.log_param("scaler", "StandardScaler")

        # Step 3: Train model
        model = train_model(X_train, X_test, y_train, y_test)

        print("Pipeline completed successfully.")


if __name__ == "__main__":
    mlflow.end_run()  # End any leftover run
    full_pipeline()
