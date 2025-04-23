#!/bin/bash

# Build Docker container
docker buildx build -t whisper . 

# Run Docker container
docker run --rm \
  --device=/dev/snd \
  --gpus all \
  --runtime=nvidia \
  --ipc=host \
  -v huggingface:/huggingface/ \
  --network=host \
  whisper &


# Save PID so we can kill it later if needed
DOCKER_PID=$!

echo "Waiting for FastAPI server to be ready..."

# Wait until port 8000 is open (max 30s)
for i in {1..30}; do
  if curl -s http://127.0.0.1:8000/docs > /dev/null; then
    echo "Server is ready!"
    break
  fi
  sleep 1
done

# Check if server didn't start
if ! curl -s http://127.0.0.1:8000/docs > /dev/null; then
  echo "Server did not start within 30 seconds."
  docker stop whisper
  exit 1
fi

trap 'echo "Stopping Docker..."; kill $DOCKER_PID' EXIT

# GPIO event loop
echo "Listening for GPIO trigger on pin 105..."
while true; do
  gpiomon -r -n 1 gpiochip0 105 | while read line; 
  do echo "Triggered"
  curl 127.0.0.1:8000/get_weather
  done
done