import torch
from chatterbox.tts import ChatterboxTTS
from pathlib import Path
import soundfile as sf
import numpy as np

print("🚀 Loading GPT-TTS safely on CPU... please wait ⏳")

# Path to your cached model
ckpt_path = Path(r"C:\Users\samxaio\.cache\huggingface\hub\models--ResembleAI--chatterbox\snapshots\05e904af2b5c7f8e482687a9d7336c5c824467d9")

# CPU-safe torch.load override
orig_load = torch.load
def cpu_load(*args, **kwargs):
    kwargs["map_location"] = torch.device("cpu")
    return orig_load(*args, **kwargs)
torch.load = cpu_load

tts = ChatterboxTTS.from_local(ckpt_path, device="cpu")
torch.load = orig_load

print("✅ Model loaded successfully on CPU!")
print("🎙️ Generating speech... this may take a few seconds")

# Actual TTS generation
text = "Hey Sameer, your Chatterbox TTS model is now loaded and running locally!"
wav = tts.generate(
    text,
    audio_prompt_path=None,   # You can pass a .wav file here to clone a voice later
    exaggeration=0.5,
    temperature=0.8,
    cfg_weight=0.5
)

# Convert and save
audio_data = wav.squeeze(0).numpy()
sf.write("output_fixed.wav", audio_data, tts.sr)

print("✅ Done! Saved voice output as output_fixed.wav 🎧")
