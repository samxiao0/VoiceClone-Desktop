import warnings
import os
import random
import numpy as np
import torch
import gradio as gr
from chatterbox.tts import ChatterboxTTS
from typing import Optional, Tuple
from datetime import datetime
import soundfile as sf
from pathlib import Path

# Silence warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Constants
DEVICE = "cpu"
MAX_TEXT_LENGTH = 2000
MAX_TEXT_SPLIT = 500
RECORDINGS_DIR = "voice_cloning_recordings"
DEFAULT_TEXT = """Once when I was six years old I saw a magnificent picture in a book..."""

# ✅ Path to your already downloaded local Chatterbox GPT-TTS model
CKPT_PATH = Path(r"C:\Users\samxaio\.cache\huggingface\hub\models--ResembleAI--chatterbox\snapshots\05e904af2b5c7f8e482687a9d7336c5c824467d9")

class CPUTTS(ChatterboxTTS):
    @classmethod
    def from_local(cls, ckpt_dir, device="cpu", **kwargs):
        original_torch_load = torch.load

        def cpu_load(*args, **kwargs):
            kwargs["map_location"] = torch.device("cpu")
            return original_torch_load(*args, **kwargs)

        torch.load = cpu_load
        try:
            model = super().from_local(ckpt_dir, device, **kwargs)
            if hasattr(model, "_model"):
                model._model.to("cpu")
            return model
        finally:
            torch.load = original_torch_load


class TTSService:
    def __init__(self):
        self.model = None

    def load_model(self) -> ChatterboxTTS:
        if self.model is None:
            print("🚀 Loading GPT-TTS model locally (CPU mode)... please wait ⏳")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = CPUTTS.from_local(CKPT_PATH, device="cpu")
                if hasattr(self.model, "_model"):
                    self.model._model.to("cpu")
            print("✅ Model loaded successfully from local cache!")
        return self.model

    @staticmethod
    def set_seed(seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def validate_inputs(text: str, audio_path: Optional[str]) -> Tuple[str, Optional[str]]:
        if not text.strip():
            raise gr.Error("🚨 Please enter some text to synthesize")
        if len(text) > MAX_TEXT_LENGTH:
            raise gr.Error(f"📜 Text too long (max {MAX_TEXT_LENGTH} characters)")
        if audio_path and not os.path.exists(audio_path):
            raise gr.Error("🔊 Reference audio file not found")
        return text, audio_path

    @staticmethod
    def save_audio(audio: Optional[Tuple[int, np.ndarray]], prefix: str = "reference") -> Optional[str]:
        if audio is None:
            return None
        sr, data = audio
        os.makedirs(RECORDINGS_DIR, exist_ok=True)
        filename = f"{RECORDINGS_DIR}/{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        sf.write(filename, data, sr)
        return filename

    @staticmethod
    def split_long_text(text: str, max_length: int = MAX_TEXT_SPLIT) -> list[str]:
        sentences = []
        current_chunk = ""
        for sentence in text.split('.'):
            if len(current_chunk) + len(sentence) < max_length:
                current_chunk += sentence + '.'
            else:
                if current_chunk:
                    sentences.append(current_chunk)
                current_chunk = sentence + '.'
        if current_chunk:
            sentences.append(current_chunk)
        return sentences

    def generate_speech(
        self,
        text: str,
        audio_prompt: Optional[Tuple[int, np.ndarray]],
        exaggeration: float,
        temperature: float,
        seed_num: int,
        cfg_weight: float
    ) -> Tuple[int, np.ndarray]:
        try:
            audio_prompt_path = self.save_audio(audio_prompt, "reference")
            text, audio_prompt_path = self.validate_inputs(text, audio_prompt_path)

            if seed_num != 0:
                self.set_seed(int(seed_num))

            model = self.load_model()

            if len(text) > MAX_TEXT_SPLIT:
                text_chunks = self.split_long_text(text)
                full_audio = []
                for chunk in text_chunks:
                    wav = model.generate(
                        chunk,
                        audio_prompt_path=audio_prompt_path,
                        exaggeration=exaggeration,
                        temperature=temperature,
                        cfg_weight=cfg_weight,
                    )
                    full_audio.append(wav.squeeze(0).numpy())
                final_audio = np.concatenate(full_audio)
                output_path = self.save_audio((model.sr, final_audio), "output")
                return model.sr, final_audio
            else:
                wav = model.generate(
                    text,
                    audio_prompt_path=audio_prompt_path,
                    exaggeration=exaggeration,
                    temperature=temperature,
                    cfg_weight=cfg_weight,
                )
                output_path = self.save_audio((model.sr, wav.squeeze(0).numpy()), "output")
                return model.sr, wav.squeeze(0).numpy()
        except Exception as e:
            raise gr.Error(f"❌ Generation failed: {str(e)}")


def create_interface() -> gr.Blocks:
    tts_service = TTSService()

    with gr.Blocks(title="🎤 VoiceClone - Unlimited Chatterbox", theme="soft") as demo:
        gr.Markdown("# 🎤 VoiceClone - Unlimited Chatterbox 🎧")
        gr.Markdown("Clone voices and generate speech with AI magic! ✨")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## ⚙️ Input Parameters")
                text_input = gr.Textbox(
                    value=DEFAULT_TEXT,
                    label=f"📝 Text to synthesize (max {MAX_TEXT_LENGTH} chars)",
                    max_lines=10,
                    placeholder="Enter your text here...",
                    interactive=True,
                )
                with gr.Group():
                    ref_audio = gr.Audio(
                        sources=["upload", "microphone"],
                        type="numpy",
                        label="🎤 Reference Audio (Wav)"
                    )
                exaggeration = gr.Slider(0.25, 2, step=0.05, value=0.5, label="🎚️ Exaggeration (Neutral = 0.5)")
                cfg_weight = gr.Slider(0.0, 1, step=0.05, value=0.5, label="⏱️ CFG/Pace Control")
                with gr.Accordion("🔧 Advanced Options", open=False):
                    seed_num = gr.Number(value=0, label="🎲 Random seed (0 = random)", precision=0)
                    temp = gr.Slider(0.05, 5, step=0.05, value=0.8, label="🌡️ Temperature (higher = more random)")
                generate_btn = gr.Button("✨ Generate Speech", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("## 🔊 Output")
                audio_output = gr.Audio(label="🎧 Generated Speech", interactive=False)
                gr.Markdown("""
                **💡 Tips:** 
                - Use clear reference audio under 10 seconds ⏱️
                - Long texts (>500 chars) will be automatically split ✂️
                - Files saved in 'voice_cloning_recordings' folder 📁
                - CPU mode may be slower ⏳
                """)

        generate_btn.click(
            fn=tts_service.generate_speech,
            inputs=[text_input, ref_audio, exaggeration, temp, seed_num, cfg_weight],
            outputs=audio_output,
            api_name="generate",
        )

    return demo


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    torch.set_default_device("cpu")
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    app = create_interface()
    app.queue(max_size=10).launch(server_name="0.0.0.0", server_port=7860, share=False)
