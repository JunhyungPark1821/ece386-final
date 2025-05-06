import sounddevice as sd
import numpy as np
import numpy.typing as npt
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Pipeline, pipeline
import sys
import time
from ollama import Client
import requests
from fastapi import FastAPI, File

app = FastAPI()

LLM_MODEL: str = "gemma3:27b"
# LLM_MODEL: str = "gemma3:1b"
client: Client = Client(
    host="http://10.1.69.213:11434/"
    # host="http://ai.dfec.xyz:11434/"
    # host="http://localhost:11434"
)

# Load Speech Recognition Model
model_id = "distil-whisper/distil-medium.en"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
pipe: Pipeline = pipeline(
    "automatic-speech-recognition",
    model=AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device),
    tokenizer=AutoProcessor.from_pretrained(model_id).tokenizer,
    feature_extractor=AutoProcessor.from_pretrained(model_id).feature_extractor,
    max_new_tokens=128,
    torch_dtype=torch_dtype,
    device=device,
)

# Picks up the USB microphone
sd.default.device = (
    "USB Audio",
    None,
)

# # Test cases
# test_cases = [ # TODO: Replace these test cases with ones for wttr.in
#     {
#         "input": "What's the weather in Rio Rancho?",
#         "expected": "rio+rancho"
#     },
#     {
#         "input": "Weather in Atlanta Airport.",
#         "expected": "atl"
#     },
#     {
#         "input": "Weather White House",
#         "expected": "~white+house"
#     },
#     {
#         "input": "Air Force Academy weather",
#         "expected": "~air+force+academy"
#     },
#     {
#         "input": "Expected weather in DEN",
#         "expected": "den"
#     },
# ]

# # Function to iterate through test cases
# def run_tests():
#     num_passed = 0

#     for i, test in enumerate(test_cases, 1):
#         raw_input = test["input"]
#         expected_output = test["expected"]

#         print(f"\nTest {i}: {raw_input}")
#         try:
#             result = llm_parse_for_wttr(raw_input).strip()
#             expected = expected_output.strip()

#             print("LLM Output  :", result)
#             print("Expected    :", expected)

#             if result == expected:
#                 print("✅ PASS")
#                 num_passed += 1
#             else:
#                 print("❌ FAIL")

#         except Exception as e:
#             print("💥 ERROR:", e)

#     print(f"\nSummary: {num_passed} / {len(test_cases)} tests passed.")


# Get the key word from user's statement
def llm_parse_for_wttr(statement: str) -> str:
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

                8. **No State:** Make sure that state or country is not included in the output if a city is given.

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


def build_pipeline(model_id: str, torch_dtype: torch.dtype, device: str) -> Pipeline:
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


def get_weather(location: str):
    """Get weather from wttr using the location from the user"""
    r = requests.get(f"https://wttr.in/{location}")

    return r.text


@app.get("/")
def read_root():
    # Whisper warmup with fake audio
    dummy_audio = np.zeros(16000, dtype=np.float32)
    _ = pipe(dummy_audio)

    # Ollama warmup
    _ = llm_parse_for_wttr("What is the weather in Paris?")
    return "FastAPI Weather Voice Service is running."


@app.get("/get_weather")
def get_weather_report():
    print("Recording...", flush=True)
    audio = record_audio()
    print("Done", flush=True)

    print("Transcribing...", flush=True)
    speech = pipe(audio)
    print("Done", flush=True)

    user_input = speech["text"]
    print("The user said: ", flush=True)
    print(user_input, flush=True)

    wttr_input = llm_parse_for_wttr(user_input).strip()
    print(wttr_input)
    report = get_weather(wttr_input)
    print(report)
    return "Report Complete. Give another location!"
