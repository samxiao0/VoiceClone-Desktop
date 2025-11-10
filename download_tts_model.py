from chatterbox.tts import ChatterboxTTS

print("🌐 Downloading GPT-TTS model from Hugging Face… this may take 10–15 minutes ⏳")

# This run only downloads and caches — it will crash at the end, ignore that
try:
    ChatterboxTTS.from_pretrained("gpt-tts")
except Exception as e:
    print(f"⚠️ Expected error after caching: {e}")

print("✅ Model files cached! You can now run test_tts_fixed.py offline.")
