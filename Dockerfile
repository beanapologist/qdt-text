FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install 'uvicorn[standard]'

# Copy the rest of the application
COPY . .

# Create a non-root user
RUN useradd -m myuser
USER myuser

# The port will be set by Cloud Run
ENV PORT=8080

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Command to run the application
CMD exec uvicorn api:app --host 0.0.0.0 --port ${PORT} 