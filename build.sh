#!/bin/bash

# Exit on error
set -e

echo "Building Docker image..."
docker build -t qdt-text:latest .

echo "Testing Docker image..."
docker run -p 8000:8000 qdt-text:latest 