# prediction-service Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app

# Expose port
EXPOSE 5003

# Run the prediction service
CMD ["python", "prediction_service.py"]
