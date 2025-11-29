# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project into container
COPY . .

# Expose MLflow UI port (optional)
EXPOSE 5000

# Default command to run training pipeline
CMD ["python", "src/pipeline_components.py"]
