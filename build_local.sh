#!/bin/bash

# Exit on error
set -e

echo "Cleaning up any existing containers..."
docker rm -f qdt-text-container 2>/dev/null || true

echo "Building Docker image..."
docker build -t qdt-text:latest .

echo "Testing Docker image..."
docker run -d --name qdt-text-container -p 8000:8000 qdt-text:latest

echo "Waiting for container to start..."
sleep 5

echo "Testing API endpoints..."
curl -s http://localhost:8000/health
echo

echo "Container is running. You can access the API at:"
echo "http://localhost:8000"
echo "API documentation at:"
echo "http://localhost:8000/docs"
echo
echo "Press Ctrl+C to stop the container"
echo "To view logs: docker logs qdt-text-container"
echo "To stop: docker stop qdt-text-container" 