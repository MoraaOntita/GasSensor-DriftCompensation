# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any required packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Install DVC
RUN pip install dvc

# Set environment variables if required
# Example: ENV MLFLOW_TRACKING_URI=http://mlflow_server:5000

# Expose the port the app runs on
EXPOSE 5000

# Define the command to run the application
CMD ["python", "app.py"]
