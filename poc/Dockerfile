# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Command to run the application (default: train the model)
#CMD ["python", "train.py", "--target", "ddr1", "--split_index", "1", "--split_type", "random", "--save_dir", "saved_models"]
# Default command to run the application (predict.py)
CMD ["python", "predict.py", "--model_file", "saved_models/chemberta_model_split1.pth", "--test_data_file", "data/test_data.csv", "--output_file", "predictions/test_predictions.csv"]
