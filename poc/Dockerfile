# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Command to run the application (modify this based on your needs)
CMD ["python", "train.py", "--target", "ddr1", "--split_index", "1", "--split_type", "random", "--save_dir", "saved_models"]
