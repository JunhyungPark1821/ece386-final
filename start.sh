#!/bin/bash

CONTAINER_NAME=whisper

# Stop existing container if still running
if docker ps -q -f name=$CONTAINER_NAME; then
  echo "Stopping existing container..."
  docker stop $CONTAINER_NAME
fi

# Remove any existing (even stopped) container named "whisper"
if docker ps -a --format '{{.Names}}' | grep -Eq "^whisper$"; then
  echo "Removing existing container named 'whisper'..."
  docker rm -f whisper
fi

# Build Docker container
docker buildx build -t $CONTAINER_NAME .

# Trap Ctrl+C and cleanup
cleanup() {
  echo "Stopping Docker container..."
  docker stop $CONTAINER_NAME
  exit 0
}
trap cleanup SIGINT

# Run Docker container with a name
docker run --rm \
  --name $CONTAINER_NAME \
  --device=/dev/snd \
  --gpus all \
  --runtime=nvidia \
  --ipc=host \
  -v huggingface:/huggingface/ \
  --network=host \
  $CONTAINER_NAME &

# Wait until port 8000 is open (max 30s)
echo "Waiting for FastAPI server to be ready..."
for i in {1..30}; do
  if curl -s http://127.0.0.1:8000/docs > /dev/null; then
    echo "Server is ready!"
    break
  fi
  sleep 1
done

# Check if server did not start
if ! curl -s http://127.0.0.1:8000/docs > /dev/null; then
  echo "Server did not start within 30 seconds."
  cleanup
fi

# GPIO event loop
echo "Listening for GPIO trigger on pin 105..."
while true; do
  gpiomon -r -n 1 gpiochip0 105 | while read -r line; do
    echo "Triggered"
    curl 127.0.0.1:8000/get_weather
  done
done
