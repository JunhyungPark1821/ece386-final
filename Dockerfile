FROM nvcr.io/nvidia/pytorch:25.02-py3-igpu
WORKDIR /app
RUN apt update && \     
    apt install -y --no-install-recommends libportaudiocpp0 libportaudio2 portaudio19-dev && \
    apt clean
RUN pip install --upgrade --no-cache-dir pip && \ 
    pip install --no-cache-dir transformers==4.49.0 accelerate==1.5.2 sounddevice ollama fastapi[standard] requests fastapi uvicorn
COPY speech_recognition.py .
ENV HF_HOME="/huggingface/"
# ENTRYPOINT ["python", "speech_recognition.py"]
# Expose the port FastAPI will run on
EXPOSE 8000
CMD ["fastapi", "run", "speech_recognition.py", "--host", "0.0.0.0", "--port", "8000"]