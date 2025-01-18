# main_app Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app

# Expose port
EXPOSE 5000

# Set environment variable for prediction service URL
ENV PREDICTION_SERVICE_URL=http://prediction-service:5003/predict

# Run the app
CMD ["python", "app.py"]
