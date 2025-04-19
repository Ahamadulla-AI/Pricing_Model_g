# Use official slim Python image
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Install dependencies first (for cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files (app.py, data, models, etc.)
COPY . .

# Create logs directory if not already in repo
RUN mkdir -p logs

# Expose the FastAPI port (match uvicorn)
EXPOSE 7860

# Run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
