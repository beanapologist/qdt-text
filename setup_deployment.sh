#!/bin/bash

# Create necessary directories
mkdir -p .github/workflows

# Ensure all files are in the correct location
echo "Setting up deployment files..."

# Copy Dockerfile if it doesn't exist
if [ ! -f Dockerfile ]; then
    echo "Creating Dockerfile..."
    cat > Dockerfile << 'EOL'
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
EOL
fi

# Create cloudbuild.yaml if it doesn't exist
if [ ! -f cloudbuild.yaml ]; then
    echo "Creating cloudbuild.yaml..."
    cat > cloudbuild.yaml << 'EOL'
steps:
  # Build the container image
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/qdt-text:$COMMIT_SHA', '.']

  # Push the container image to Container Registry
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/qdt-text:$COMMIT_SHA']

  # Deploy container image to Cloud Run
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: gcloud
    args:
      - 'run'
      - 'deploy'
      - 'qdt-text'
      - '--image'
      - 'gcr.io/$PROJECT_ID/qdt-text:$COMMIT_SHA'
      - '--region'
      - 'us-central1'
      - '--platform'
      - 'managed'
      - '--allow-unauthenticated'
      - '--memory'
      - '512Mi'
      - '--cpu'
      - '1'
      - '--min-instances'
      - '0'
      - '--max-instances'
      - '10'
      - '--port'
      - '8000'

images:
  - 'gcr.io/$PROJECT_ID/qdt-text:$COMMIT_SHA'
EOL
fi

# Create .dockerignore if it doesn't exist
if [ ! -f .dockerignore ]; then
    echo "Creating .dockerignore..."
    cat > .dockerignore << 'EOL'
# Git
.git
.gitignore

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
.venv
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
*.pth
*.pkl
*.log
tests/
.github/
cloudbuild.yaml
EOL
fi

# Make the script executable
chmod +x setup_deployment.sh

echo "Setup complete! Please run:"
echo "1. git add ."
echo "2. git commit -m 'Add deployment configuration'"
echo "3. git push origin main" 