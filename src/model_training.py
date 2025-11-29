from pipeline_components import (
    load_data,
    preprocess_data,
    train_model,
    evaluate_model
)


def run_pipeline():
    df = load_data("data/raw/california_housing.csv")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    model = train_model(X_train, y_train)
    mse, r2 = evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    run_pipeline()
