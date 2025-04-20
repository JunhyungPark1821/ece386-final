#!/bin/bash

docker buildx build . -t whisper
docker run -it --rm --device=/dev/snd --runtime=nvidia --ipc=host -v huggingface:/huggingface/ whisper