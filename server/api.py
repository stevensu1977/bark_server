from model import load
import soundfile as sf
from pydub import AudioSegment
import numpy as np
from io import BytesIO
import time
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from typing import Optional
from pydantic import BaseModel
import uvicorn
import os

import scipy
import nltk
from bark import generate_audio, SAMPLE_RATE

host = os.getenv("HOST", "0.0.0.0")
port = os.getenv("PORT", "8000")

port = int(port)

SUPPORT_SPEAKERS=["v2/en_speaker_6","v2/en_speaker_9"]

def predict_long_form(processor,model,txt,voice_preset=None):
    sentences = nltk.sent_tokenize(txt.replace("\n", " ").strip())
    SPEAKER = "v2/en_speaker_6"
    if voice_preset in SUPPORT_SPEAKERS:
        SPEAKER = voice_preset
    
    silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence

    print(f"{voice_preset=}")
    pieces = []
    for sentence in sentences:
        audio_array = generate_audio(sentence, history_prompt=SPEAKER)
        pieces += [audio_array, silence.copy()]

   
    #pieces = model.generate(**inputs, do_sample=True)
    
    # audio_array = np.concatenate([i.cpu().numpy()[:l] for i,l in zip(pieces[0],pieces[1])])
    #scipy.io.wavfile.write("../outputs/example.wav",rate=SAMPLE_RATE, data=np.concatenate(pieces))

    
    # Now we need to move our tensors back to the CPU and convert them to a NumPy array
    #speech = speech_values.cpu().numpy().squeeze()

    # The soundfile library requires us to convert our array of float16 numbers
    # into an array of int16 numbers.
    #scaled_speech = np.int16(speech * 32767)

    wav = BytesIO()
    sf.write(wav, np.concatenate(pieces), SAMPLE_RATE, format="WAV", subtype="PCM_16")


    # WAV files are huge, so we're going to use pydub to convert it to a much
    # smaller mp3 "file"
    audio = AudioSegment.from_wav(wav)
    dur_ms = len(audio)

    mp3 = BytesIO()
    audio.export(mp3, format="mp3")

    return mp3, dur_ms
    #return "",""
    

    

def predict(processor, model, text, voice_preset=None):
    # We need to convert our text into something the model can understand,
    # using the processor. We're going to have it return PyTorch tensors.
    inputs = processor(text, voice_preset=voice_preset, return_tensors="pt")

    # Now we need to move these tensors to the GPU, so they can be processed
    # by the model.
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # The model generates more tensors that can be decoded into audio.
    speech_values = model.generate(**inputs, do_sample=True)

    # For interpreting the tensors back into audio, we need
    # to know the sampling_rate that was used to generate it.
    sampling_rate = model.generation_config.sample_rate

    # Now we need to move our tensors back to the CPU and convert them to a NumPy array
    speech = speech_values.cpu().numpy().squeeze()

    # The soundfile library requires us to convert our array of float16 numbers
    # into an array of int16 numbers.
    scaled_speech = np.int16(speech * 32767)

    # We don't want to actually write to the file system, so we'll create
    # a file-like BytesIO object, and write the wav to that
    wav = BytesIO()
    sf.write(wav, scaled_speech, sampling_rate, format="WAV", subtype="PCM_16")

    # WAV files are huge, so we're going to use pydub to convert it to a much
    # smaller mp3 "file"
    audio = AudioSegment.from_wav(wav)
    dur_ms = len(audio)

    mp3 = BytesIO()
    audio.export(mp3, format="mp3")

    return mp3, dur_ms


class GenerateRequest(BaseModel):
    # Add your input parameters here
    text: str
    voice_preset: Optional[str] = None


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/hc")
async def health_check():
    return "OK"


@app.post("/generate", response_class=StreamingResponse)
def generate(req: GenerateRequest = Body(...)):
    try:
        #mp3, dur_ms = predict(processor, model, req.text, req.voice_preset)
        mp3, dur_ms = predict_long_form(processor,model, req.text, req.voice_preset)
        print(f"Voice duration {dur_ms}/ms")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(mp3, media_type="audio/mpeg")


if __name__ == "__main__":
    print("Loading model...", flush=True)
    start = time.perf_counter()
    processor, model = load()
    end = time.perf_counter()
    print(f"Model loaded in {end - start:.2f}s", flush=True)

    print("Generating test audio...", flush=True)
    start = time.perf_counter()
    test_text = "Hello, my name is Suno and I like to Bark!"
    mp3, dur_ms = predict(processor, model, test_text)
    end = time.perf_counter()
    print(
        f"Test audio ({dur_ms / 1000:.2f}s) generated in {end - start:.2f}s", flush=True
    )
    uvicorn.run(app, host=host, port=port)
