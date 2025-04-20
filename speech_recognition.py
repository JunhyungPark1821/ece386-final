import sounddevice as sd
import numpy as np
import numpy.typing as npt
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Pipeline, pipeline
import sys
import time
from ollama import Client

LLM_MODEL: str = "gemma3:27b"
# LLM_MODEL: str = "gemma3:1b"
client: Client = Client(
  host='http://ai.dfec.xyz:11434/'
  # host="http://localhost:11434"
)

def llm_parse_for_wttr(statement):
  response = client.chat(
    
      messages=[
          {
              "role": "system",
              "content": """ 
                # Task: Convert Natural Language Weather Requests to wttr.in Format

                # Overview
                You will process user requests for weather information and convert them into the specific format required by wttr.in. This involves accurately identifying the type of location requested (city, airport, 3-letter airport code, or geographic location) and formatting it accordingly.

                # Input Format
                The user will provide a sentence containing a location. This location can be:
                - A city name (e.g., Seoul, Washington D.C., Montgomery, Denver).
                - An airport name (e.g., Atlanta Airport, Denver Airport, Montgomery Airport).
                - A 3-letter IATA airport code (e.g., atl, lax, mgm, den).
                - A geographic location (e.g., White House, USAFA, Eiffel Tower, Space Needle).

                # Output Requirements
                You must generate a single line of text, entirely in lowercase, representing the location in the format suitable for wttr.in. Follow these rules:

                1. **Identify Location Type:** Determine if the user's request refers to a city, airport name, 3-letter airport code, or geographic location.

                2. **City or Geographic Location (Multiple Words):** If the identified location is a city or geographic location consisting of two or more words, replace all spaces with a plus sign (`+`).

                3. **Airport Name:** If the request includes the word "airport" (case-insensitive), extract the corresponding 3-letter IATA airport code and output it in lowercase. You will need to have access to an internal mapping of airport names to their 3-letter codes.

                4. **3-Letter Airport Code:** If the input is already a 3-letter airport code (regardless of its original casing), output it directly in lowercase.

                5. **Geographic Location (Not Airport/Code/City):** If the request is a recognized geographic location that doesn't fall into the categories above, prepend a tilde character (`~`) to the formatted name (with spaces replaced by `+` if it's multi-word).

                6. **Single-Word City:** If the request is a single-word city name, output it directly in lowercase.

                7. **No Newlines:** Ensure that the output contains no newline characters (`\n`).

                # Examples

                **Input:** London
                **Output:** london
                **Reason:** "London" is a single-word city.

                **Input:** Salt Lake City
                **Output:** salt+lake+city
                **Reason:** "Salt Lake City" is a multi-word city, so spaces are replaced with "+".

                **Input:** Munich International Airport
                **Output:** muc
                **Reason:** The input contains "Airport," so the corresponding 3-letter airport code "muc" is output in lowercase.

                **Input:** Hamburg Airport
                **Output:** ham
                **Reason:** The input contains "Airport," so the corresponding 3-letter airport code "ham" is output in lowercase.

                **Input:** HAM
                **Output:** ham
                **Reason:** "HAM" is recognized as a 3-letter airport code and is output in lowercase.

                **Input:** Eiffel Tower
                **Output:** ~eiffel+tower
                **Reason:** "Eiffel Tower" is a multi-word geographic location (not a city or airport), so it's prefixed with "~" and spaces are replaced with "+".

                **Input:** White House
                **Output:** ~white+house
                **Reason:** "White House" is a multi-word geographic location.

                **Input:** USAFA
                **Output:** ~usafa
                **Reason:** "USAFA" is a single-word geographic location.

                **Important Note:** For rules 3 and 4 (airport names and 3-letter codes), you will need to have a mechanism to accurately map airport names to their IATA codes. This prompt assumes you have access to or can build such a mapping.
                """,
          },
          {
              "role": "user",
              "content": statement,
          },
      ],
      model=LLM_MODEL,
  )

  return response.message.content

def record_audio(duration_seconds: int = 10) -> npt.NDArray:
    """Record duration_seconds of audio from default microphone.
    Return a single channel numpy array."""
    sample_rate = 16000  # Hz
    samples = int(duration_seconds * sample_rate)
    # Will use default microphone; on Jetson this is likely a USB WebCam
    audio = sd.rec(samples, samplerate=sample_rate, channels=1, dtype=np.float32)
    # Blocks until recording complete
    sd.wait()
    # Model expects single axis
    return np.squeeze(audio, axis=1)

def build_pipeline(
    model_id: str, torch_dtype: torch.dtype, device: str
) -> Pipeline:
    """Creates a Hugging Face automatic-speech-recognition pipeline on the given device."""
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        torch_dtype=torch_dtype,
        device=device,
    )
    return pipe

if __name__ == "__main__":
    # # Get model as argument, default to "distil-whisper/distil-medium.en" if not given
    # model_id = sys.argv[1] if len(sys.argv) > 1 else "distil-whisper/distil-medium.en"
    # print("Using model_id {model_id}")
    # # Use GPU if available
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    # print(f"Using device {device}.")

    # print("Building model pipeline...")
    # pipe = build_pipeline(model_id, torch_dtype, device)
    # print(type(pipe))
    # print("Done")

    # print("Recording...")
    # audio = record_audio()
    # print("Done")

    # print("Transcribing...")
    # start_time = time.time_ns()
    # speech = pipe(audio)
    # end_time = time.time_ns()
    # print("Done")

    # print(speech)
    # print(f"Transcription took {(end_time-start_time)/1000000000} seconds")
    result = llm_parse_for_wttr("Denver Airport").strip()
    print(result)