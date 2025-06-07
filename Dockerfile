FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create a non-root user
RUN useradd -m myuser
USER myuser

# The port will be set by Cloud Run
ENV PORT=8000

# Expose port 8000 for Cloud Run
EXPOSE 8000

# Command to run the application
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT} 