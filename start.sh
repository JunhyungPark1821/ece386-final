#!/bin/bash

docker buildx build . -t whisper

while true
do
  gpiomon -r -n 1 gpiochip0 105 | while read line; do
    echo "Triggered"
    docker run -t --rm --device=/dev/snd --gpus all --runtime=nvidia --ipc=host -v huggingface:/huggingface/ whisper
  done
done