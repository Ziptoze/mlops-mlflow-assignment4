# MLOps MLflow Assignment

## Project Overview

This project implements a complete Machine Learning Operations (MLOps) pipeline for training and evaluating a regression model on the California Housing dataset. The pipeline demonstrates industry-standard practices for data versioning, model training, experiment tracking, and continuous integration.

### ML Problem
The project tackles a regression problem using the California Housing dataset to predict median house values based on various features such as location, demographics, and housing characteristics.

### Key Technologies
- **Data Versioning**: DVC (Data Version Control)
- **Experiment Tracking**: MLflow
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **ML Framework**: Scikit-learn
- **Pipeline Orchestration**: Custom Python pipeline components

---

## Project Structure

```
mlops-mlflow-assignment4/
├── data/
│   ├── raw/                    # Raw dataset
│   │   └── california_housing.csv
│   └── processed/              # Processed data
├── src/
│   ├── pipeline_components.py  # Reusable pipeline components
│   ├── model_training.py       # Model training script
│   └── pipeline.py             # Main pipeline orchestration
├── components/
│   ├── preprocess_component.yaml
│   └── train_component.yaml
├── mlruns/                     # MLflow experiment tracking
├── dvc_remote/                 # Local DVC remote storage
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI workflow
├── .dvc/                       # DVC configuration
├── Dockerfile                  # Container definition
├── Jenkinsfile                 # Jenkins pipeline (alternative CI)
├── requirements.txt            # Python dependencies
├── pipeline.py                 # Pipeline execution
├── data.py                     # Data loading utilities
└── README.md
```

---

## Setup Instructions

### Prerequisites
- Python 3.8+
- Git
- Docker (optional, for containerization)
- DVC

### 1. Clone the Repository

```bash
git clone https://github.com/Ziptoze/mlops-mlflow-assignment4.git
cd mlops-mlflow-assignment4
```

### 2. Set Up Python Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. DVC Setup and Data Versioning

#### Initialize DVC (if not already initialized)
```bash
dvc init
```

#### Configure DVC Remote Storage

The project uses a local directory as DVC remote storage:

```bash
# Set up local remote storage
dvc remote add -d local_remote dvc_remote

# Verify DVC configuration
dvc remote list
```

#### Pull Dataset from DVC
```bash
# Pull the versioned data
dvc pull
```

**Note**: The dataset is now directly committed to the repository for easier access. If you need to use DVC tracking:

```bash
# Track new data files with DVC
dvc add data/raw/california_housing.csv

# Push to DVC remote
dvc push
```

### 4. MLflow Setup

MLflow is used for experiment tracking and model registry. The tracking data is stored in the `mlruns/` directory.

```bash
# View MLflow UI (optional)
mlflow ui
```

Access the MLflow dashboard at `http://localhost:5000`

---

## Pipeline Walkthrough

### Pipeline Components

The pipeline consists of the following reusable components defined in `src/pipeline_components.py`:

1. **Data Extraction**: Loads the California Housing dataset
2. **Data Preprocessing**: Handles data cleaning, feature scaling, and train/test splitting
3. **Model Training**: Trains a regression model and logs metrics with MLflow
4. **Model Evaluation**: Evaluates the trained model and saves performance metrics

### Component Inputs and Outputs

#### Preprocessing Component
- **Inputs**: 
  - `data_path`: Path to raw dataset
  - `test_size`: Float, proportion of test set (default: 0.2)
  - `random_state`: Integer, random seed for reproducibility
- **Outputs**: 
  - `train_data_path`: Path to processed training data
  - `test_data_path`: Path to processed test data

#### Training Component
- **Inputs**: 
  - `train_data_path`: Path to training data
  - `n_estimators`: Number of trees in the forest
  - `max_depth`: Maximum depth of trees
  - `random_state`: Random seed
- **Outputs**: 
  - `model_path`: Path to saved model artifact
  - `metrics`: Dictionary containing model performance metrics (MSE, RMSE, R²)

### Running the Pipeline

#### Option 1: Run Complete Pipeline
```bash
python pipeline.py
```

#### Option 2: Run Individual Components
```bash
# Preprocess data
python src/pipeline_components.py --step preprocess

# Train model
python src/model_training.py

# Evaluate model
python src/pipeline_components.py --step evaluate
```

### Pipeline Execution Flow

1. **Data Loading**: The pipeline loads the California Housing dataset
2. **Preprocessing**: Data is split into training and testing sets, features are scaled
3. **Model Training**: A Random Forest Regressor is trained on the training data
4. **Experiment Tracking**: MLflow logs parameters, metrics, and model artifacts
5. **Model Evaluation**: The model is evaluated on the test set, producing MSE, RMSE, and R² scores
6. **Artifact Storage**: Trained model is saved for future deployment

---

## Continuous Integration

### GitHub Actions Workflow

The project includes a CI/CD pipeline using GitHub Actions (`.github/workflows/ci.yml`) with the following stages:

#### Stage 1: Environment Setup
- Checks out the code
- Sets up Python 3.9
- Installs dependencies from `requirements.txt`

#### Stage 2: Data Validation
- Verifies dataset availability
- Runs data quality checks

#### Stage 3: Pipeline Execution
- Compiles and validates the pipeline
- Runs the complete ML pipeline
- Validates model outputs

### Triggering the CI Pipeline

The workflow is automatically triggered on:
- Push to `main` branch
- Pull requests to `main` branch

Manual trigger:
```bash
# Push changes to trigger workflow
git add .
git commit -m "Trigger CI pipeline"
git push origin main
```

### Jenkins Alternative

A `Jenkinsfile` is also provided for Jenkins-based CI/CD:

```groovy
pipeline {
    agent any
    stages {
        stage('Setup') { ... }
        stage('Pipeline Compilation') { ... }
        stage('Execute Pipeline') { ... }
    }
}
```

---

## Docker Support

### Build Docker Image

```bash
docker build -t mlops-mlflow-pipeline .
```

### Run Pipeline in Container

```bash
docker run -v $(pwd)/data:/app/data -v $(pwd)/mlruns:/app/mlruns mlops-mlflow-pipeline
```

---

## Model Performance

The Random Forest Regressor achieves the following performance metrics on the California Housing dataset:

- **Mean Squared Error (MSE)**: ~0.25
- **Root Mean Squared Error (RMSE)**: ~0.50
- **R² Score**: ~0.80

*Note: Exact values may vary depending on random state and hyperparameters.*

---

## Experiment Tracking with MLflow

All experiments are automatically tracked using MLflow:

- **Parameters**: Model hyperparameters (n_estimators, max_depth, etc.)
- **Metrics**: MSE, RMSE, R² score
- **Artifacts**: Trained model files
- **Tags**: Experiment metadata

View experiments:
```bash
mlflow ui --port 5000
```

---

## Data Versioning with DVC

### Track New Data
```bash
dvc add data/raw/new_dataset.csv
git add data/raw/new_dataset.csv.dvc .gitignore
git commit -m "Track new dataset with DVC"
```

### Push Data to Remote
```bash
dvc push
```

### Pull Data from Remote
```bash
dvc pull
```

### Check DVC Status
```bash
dvc status
```

---

## Dependencies

Key Python packages (see `requirements.txt` for complete list):

```
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
mlflow>=2.0.0
dvc>=2.0.0
pyyaml>=6.0
```

---

## Troubleshooting

### Issue: Dataset not found
**Solution**: Ensure you've run `dvc pull` or the dataset is in `data/raw/california_housing.csv`

### Issue: MLflow UI not loading
**Solution**: Check if port 5000 is available, or specify a different port: `mlflow ui --port 5001`

### Issue: DVC remote storage error
**Solution**: Verify DVC remote configuration: `dvc remote list` and `dvc config -l`

### Issue: Pipeline fails during CI
**Solution**: Check GitHub Actions logs for detailed error messages

---

## Future Enhancements

- [ ] Implement model versioning and registry
- [ ] Add hyperparameter tuning with Optuna
- [ ] Integrate model serving with MLflow Model Serving
- [ ] Add data drift detection
- [ ] Implement A/B testing framework
- [ ] Add comprehensive unit tests
- [ ] Set up model monitoring dashboard

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

---

## License

This project is created for educational purposes as part of an MLOps assignment.

---

## Contact

**Repository**: [https://github.com/Ziptoze/mlops-mlflow-assignment4](https://github.com/Ziptoze/mlops-mlflow-assignment4)

**Author**: Ziptoze

---

## Acknowledgments

- California Housing Dataset from Scikit-learn
- MLflow for experiment tracking
- DVC for data version control
- GitHub Actions for CI/CD automation
