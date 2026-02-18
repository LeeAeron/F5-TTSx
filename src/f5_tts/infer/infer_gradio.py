# ruff: noqa: E402
# Above allows ruff to ignore E402: module level import not at top of file
import asyncio
import sys
import logging
import threading
import time
import webbrowser
import uvicorn


if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import gc
import json
import os
import re
import tempfile
import requests
from collections import OrderedDict
from functools import lru_cache
from importlib.resources import files
import zipfile

import click
import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torchaudio
from cached_path import cached_path
from transformers import AutoModelForCausalLM, AutoTokenizer

os.makedirs("outputs", exist_ok=True)

import subprocess
import soundfile as sf
import tempfile

from datetime import datetime

def save_audio(final_wave, final_sample_rate, fmt="wav", used_seed=None):
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp_wav.name, final_wave, final_sample_rate)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    seed_str = str(used_seed) if used_seed is not None else "noseed"
    out_filename = f"tts_{timestamp}_{seed_str}.{fmt}"
    out_path = os.path.join("outputs", out_filename)

    subprocess.run([
        "ffmpeg", "-y", "-i", tmp_wav.name, out_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return out_path


try:
    import spaces

    USING_SPACES = True
except ImportError:
    USING_SPACES = False


def gpu_decorator(func):
    if USING_SPACES:
        return spaces.GPU(func)
    else:
        return func


from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    save_spectrogram,
    tempfile_kwargs,
)
from f5_tts.model import DiT, UNetT


import os
import requests
import time

def safe_path(local_path: str, remote_url: str) -> str:
    """
    Checks for the existence of a local file.
    If it doesn't exist, it downloads remote_url to local_path and returns the path.
    Shows progress, elapsed time, and estimated remaining time.
    """
    if os.path.exists(local_path):
        return local_path

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    print(f"[INFO] Downloading {remote_url} -> {local_path}")
    r = requests.get(remote_url, stream=True)
    r.raise_for_status()

    total_size = int(r.headers.get("Content-Length", 0))
    downloaded = 0
    start = time.time()

    with open(local_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)

                if total_size > 0:
                    elapsed = time.time() - start
                    speed = downloaded / elapsed  # bytes per second
                    remaining = (total_size - downloaded) / speed if speed > 0 else 0
                    percent = downloaded / total_size * 100
                    print(
                        f"\r[INFO] {percent:.2f}% | "
                        f"Elapsed: {elapsed:.1f}s | "
                        f"Remaining: {remaining:.1f}s",
                        end=""
                    )

    print(f"\n[INFO] Download complete: {local_path}")

    return local_path


# --- basic settings ---
DEFAULT_TTS_MODEL = "F5-TTS_v1"
tts_model_choice = DEFAULT_TTS_MODEL

DEFAULT_TTS_MODEL_CFG = [
    safe_path(
        "models/F5TTS_v1_Base/F5TTS_v1_Base_1250000.safetensors",
        "https://huggingface.co/datasets/LeeAeron/F5TTSx/resolve/main/models/F5TTS_v1_Base/F5TTS_v1_Base_1250000.safetensors"
    ),
    safe_path(
        "models/F5TTS_v1_Base/vocab.txt",
        "https://huggingface.co/datasets/LeeAeron/F5TTSx/resolve/main/models/F5TTS_v1_Base/vocab.txt"
    ),
    json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2,
                    text_dim=512, conv_layers=4)),
]


# load models
vocoder = load_vocoder()


def load_f5tts():
    ckpt_path = str(cached_path(DEFAULT_TTS_MODEL_CFG[0]))
    F5TTS_model_cfg = json.loads(DEFAULT_TTS_MODEL_CFG[2])
    return load_model(DiT, F5TTS_model_cfg, ckpt_path)


def load_e2tts():
    ckpt_path = str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors"))
    E2TTS_model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4, text_mask_padding=False, pe_attn_head=1)
    return load_model(UNetT, E2TTS_model_cfg, ckpt_path)


def load_custom(ckpt_path: str, vocab_path="", model_cfg=None):
    ckpt_path, vocab_path = ckpt_path.strip(), vocab_path.strip()
    if ckpt_path.startswith("hf://"):
        ckpt_path = str(cached_path(ckpt_path))
    if vocab_path.startswith("hf://"):
        vocab_path = str(cached_path(vocab_path))
    if model_cfg is None:
        model_cfg = json.loads(DEFAULT_TTS_MODEL_CFG[2])
    elif isinstance(model_cfg, str):
        model_cfg = json.loads(model_cfg)
    return load_model(DiT, model_cfg, ckpt_path, vocab_file=vocab_path)


F5TTS_ema_model = load_f5tts()
E2TTS_ema_model = load_e2tts() if USING_SPACES else None
custom_ema_model, pre_custom_path = None, ""

chat_model_state = None
chat_tokenizer_state = None


@gpu_decorator
def chat_model_inference(messages, model, tokenizer):
    """Generate response using Qwen"""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
    )

    generated_ids = [
        output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


@gpu_decorator
def load_text_from_file(file):
    if file:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read().strip()
    else:
        text = ""
    return gr.update(value=text)


# --- Accentuation Support ---
try:
    from ruaccent import RUAccent
    accent_model = RUAccent()
    accent_model.load()
except Exception as e:
    print(f"Failed to initialize RUAccent: {e}")
    accent_model = None

import unicodedata, re, yaml, os
from pathlib import Path
import chardet

# --- External dictionaries ---
def _sanitize_pair(k, v):
    if not isinstance(k, str) or not isinstance(v, str):
        return None
    k = k.lstrip("*").strip()
    v = v.strip()
    if not k or not v:
        return None
    return k, v

def load_dic_file(path: Path) -> dict[str, str]:
    fixes = {}
    try:
        raw_bytes = path.read_bytes()
        detected = chardet.detect(raw_bytes)
        encoding = detected["encoding"] or "utf-8"
        with open(path, "r", encoding=encoding, errors="ignore") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                left, right = line.split("=", 1)
                fixes[left.strip()] = right.strip()
    except Exception as e:
        print(f"Failed to load dic file {path}: {e}")
    return fixes

def load_dict_file(path: Path) -> dict[str, str]:
    fixes = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                left, right = line.split("=", 1)
                pair = _sanitize_pair(left, right)
                if pair:
                    fixes[pair[0]] = pair[1]
    except Exception as e:
        print(f"Failed to load dict file {path}: {e}")
    return fixes

def load_all_dicts(folder: Path) -> dict[str, str]:
    all_fixes = {}
    if folder.is_dir():
        for name in sorted(os.listdir(folder)):
            file_path = folder / name
            if name.lower().endswith(".dict"):
                all_fixes.update(load_dict_file(file_path))
            elif name.lower().endswith(".dic"):
                all_fixes.update(load_dic_file(file_path))
    return all_fixes

def load_custom_accents() -> dict[str, str]:
    yaml_fixes = {}
    path = Path("accent_fixes.yaml")
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    for k, v in data.items():
                        pair = _sanitize_pair(k, v)
                        if pair:
                            yaml_fixes[pair[0]] = pair[1]
        except Exception as e:
            print(f"Failed to load accent_fixes.yaml: {e}")
    dict_folder = Path("./dicts")
    dict_fixes = load_all_dicts(dict_folder)
    combined = {**yaml_fixes, **dict_fixes}
    print(f"Loaded {len(combined)} custom accent fixes (YAML + dicts)")
    return combined

CUSTOM_ACCENTS = load_custom_accents()

def apply_custom_fixes(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    items = [(k, v) for k, v in CUSTOM_ACCENTS.items() if isinstance(k, str) and isinstance(v, str)]
    items.sort(key=lambda kv: len(kv[0]), reverse=True)
    for wrong, correct in items:
        text = re.sub(re.escape(wrong), correct, text, flags=re.IGNORECASE)
    return text

# --- RUAccent + convert_accent_to_plus ---
def convert_accent_to_plus(text: str) -> str:
    replacements = {
        'а́': '+а', 'А́': '+А',
        'е́': '+е', 'Е́': '+Е',
        'ё́': '+йо', 'Ё́': '+ЙО',
        'и́': '+и', 'И́': '+И',
        'о́': '+о', 'О́': '+О',
        'у́': '+у', 'У́': '+У',
        'ы́': '+ы', 'Ы́': '+Ы',
        'э́': '+э', 'Э́': '+Э',
        'ю́': '+ю', 'Ю́': '+Ю',
        'я́': '+я', 'Я́': '+Я',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def preprocess_text(text: str) -> str:
    if accent_model is not None:
        try:
            text = accent_model.process_all(text)  # RUAccent → +markers
        except Exception as e:
            print(f"RUAccent failed: {e}")
    text = convert_accent_to_plus(text)          # normal logic
    text = apply_custom_fixes(text)              # dictionary + YAML
    return text


from fastapi import FastAPI
from fastapi.responses import JSONResponse
import sys, os, asyncio, logging

logger = logging.getLogger(__name__)

api_app = FastAPI()

@api_app.post("/api/restart", tags=["System"])
async def restart_engine_endpoint():
    logger.warning("⚠ Restart request received via /api/restart endpoint.")

    async def _delayed_restart():
        await asyncio.sleep(2)
        python = sys.executable
        os.execl(python, python, *sys.argv)

    asyncio.create_task(_delayed_restart())
    return JSONResponse({"status": "restarting", "message": "Server restarting..."})


@lru_cache(maxsize=1000)  # NOTE. need to ensure params of infer() hashable
@gpu_decorator
def infer(
    ref_audio_orig,
    ref_text,
    gen_text,
    model,
    remove_silence,
    seed,
    cross_fade_duration=0.0,
    nfe_step=16,
    speed=1,
    use_accent=True,
    target_rms=0.1,
    cfg_strength=1.0,
    show_info=gr.Info,
):
    if not ref_audio_orig:
        gr.Warning("Please provide reference audio.")
        return gr.update(), gr.update(), ref_text

    # Set inference seed
    if seed < 0 or seed > 2**31 - 1:
        gr.Warning("Seed must in range 0 ~ 2147483647. Using random seed instead.")
        seed = np.random.randint(0, 2**31 - 1)
    torch.manual_seed(seed)
    used_seed = seed

    if not gen_text.strip():
        gr.Warning("Please enter text to generate or upload a text file.")
        return gr.update(), gr.update(), ref_text

    if use_accent:
        gen_text = preprocess_text(gen_text)

    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=show_info)

    ema_model = None

    # universal check: string or tuple
    if isinstance(model, tuple):
        model_type = model[0]
    else:
        model_type = model

    if model_type == DEFAULT_TTS_MODEL:  # "F5-TTS_v1"
        ema_model = F5TTS_ema_model

    elif model_type == "E2-TTS":
        global E2TTS_ema_model
        if E2TTS_ema_model is None:
            show_info("Loading E2-TTS model...")
            E2TTS_ema_model = load_e2tts()
        ema_model = E2TTS_ema_model

    elif model_type in [
        "Custom",
        "Misha24-10_v2",
        "Misha24-10_v4",
        "ESpeech-TTS-1_podcaster",
        "ESpeech-TTS-1_RL-V1",
        "ESpeech-TTS-1_RL-V2",
        "ESpeech-TTS-1_SFT-95K",
        "ESpeech-TTS-1_SFT-256K"
    ]:
        assert not USING_SPACES, "Only official checkpoints allowed in Spaces."
        global custom_ema_model, pre_custom_path
        ckpt, vocab, cfg = model[1], model[2], model[3]
        if pre_custom_path != ckpt:
            show_info(f"Loading {model_type} TTS model...")
            custom_ema_model = load_custom(ckpt, vocab_path=vocab, model_cfg=cfg)
            pre_custom_path = ckpt
        ema_model = custom_ema_model

    if ema_model is None:
        raise ValueError(f"Unknown model type: {model_type}")

    final_wave, final_sample_rate, combined_spectrogram = infer_process(
        ref_audio,
        ref_text,
        gen_text,
        ema_model,
        vocoder,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        speed=speed,
        target_rms=target_rms,
        cfg_strength=cfg_strength,
        show_info=show_info,
        progress=gr.Progress(),
    )

    # Remove silence
    if remove_silence:
        with tempfile.NamedTemporaryFile(suffix=".wav", **tempfile_kwargs) as f:
            temp_path = f.name
        try:
            sf.write(temp_path, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
        finally:
            os.unlink(temp_path)
        final_wave = final_wave.squeeze().cpu().numpy()

    # Save the spectrogram
    with tempfile.NamedTemporaryFile(suffix=".png", **tempfile_kwargs) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
    save_spectrogram(combined_spectrogram, spectrogram_path)

    final_wave = (final_wave * 32767).astype(np.int16)

    return (final_sample_rate, final_wave), spectrogram_path, ref_text, used_seed


# reference voices download logic
ROOT_DIR = os.environ.get("ROOT", os.getcwd())
CACHE_DIR = os.environ.get("CACHE", os.path.join(ROOT_DIR, "cache"))
VOICES_DIR = os.path.join(ROOT_DIR, "voices")

os.makedirs(VOICES_DIR, exist_ok=True)

import os

def get_voice_choices():
    files = []
    for f in os.listdir(VOICES_DIR):
        if f.lower().endswith((".wav", ".mp3", ".aac", ".m4a", ".m4b", ".ogg", ".flac", ".opus")):
            try:
                fixed = f.encode("cp1251").decode("utf-8")
            except UnicodeError:
                fixed = f
            files.append(fixed)
    return ["-NONE-"] + files

def download_and_extract(url):
    archive_path = os.path.join(CACHE_DIR, "voices_tmp.zip")

    if os.path.exists(archive_path):
        os.remove(archive_path)

    r = requests.get(url, stream=True)
    if r.status_code != 200:
        print(f"Error while downloading: {r.status_code}")
        return gr.update(choices=get_voice_choices(), value="-NONE-")

    with open(archive_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(VOICES_DIR)

    os.remove(archive_path)

    return gr.update(choices=get_voice_choices(), value="-NONE-")


available_voices = [
    f for f in os.listdir(VOICES_DIR)
    if os.path.isfile(os.path.join(VOICES_DIR, f)) and f.endswith((".wav", ".mp3", ".aac", ".m4a", ".m4b", ".ogg", ".flac", ".opus"))
]

voice_choices = ["-NONE-"] + available_voices

def clear_audio():
    return None

def set_voice_file(selected_voice):
    if selected_voice and selected_voice != "-NONE-":
        file_path = os.path.join(VOICES_DIR, selected_voice)
        if os.path.isfile(file_path):
            return os.path.abspath(file_path)
        else:
            return None
    return None


#batched TTS tab
with gr.Blocks() as app_tts:
    gr.Markdown("# Batched TTS")

    with gr.Row():
        # Left column: voice and audio selection
        with gr.Column(scale=3):
            voice_selector = gr.Dropdown(
                choices=voice_choices,
                label="Pre-defined voices (folder 'voices')",
                value="-NONE-",
            )
            # Update btn + download btn
            with gr.Row():
                refresh_btn = gr.Button("🔄", elem_classes="square-btn")
                download_btn = gr.Button("⬇️ Download Voices")

            ref_audio_input = gr.Audio(label="Reference Audio", type="filepath")

        # Right column: text field
        with gr.Column(scale=3):
            ref_text_input = gr.Textbox(
                label="Reference Text",
                info="Leave blank to automatically transcribe the reference audio. "
                     "If you enter text or upload a file, it will override automatic transcription.",
                lines=2,
            )
            ref_text_file = gr.File(
                label="Load Reference Text from File (.txt)",
                file_types=[".txt"],
            )

    refresh_btn.click(
        lambda: gr.update(choices=get_voice_choices(), value="-NONE-"),
        inputs=None,
        outputs=voice_selector,
    )

    download_btn.click(
        lambda: download_and_extract("https://huggingface.co/datasets/LeeAeron/F5TTSx/resolve/main/ref_voices.zip"),
        inputs=None,
        outputs=voice_selector,
    )

    voice_selector.change(
        lambda: (None, ""),
        inputs=None,
        outputs=[ref_audio_input, ref_text_input],
    ).then(
        set_voice_file,
        inputs=[voice_selector],
        outputs=[ref_audio_input],
    )

    format_selector = gr.Dropdown(
        choices=["wav", "mp3", "aac", "m4a", "m4b", "ogg", "flac", "opus"],
        value="wav",
        label="Output Format"
    )

    with gr.Row():
        gen_text_input = gr.Textbox(
            label="Text to Generate",
            lines=10,
            max_lines=40,
            scale=4,
        )
        gen_text_file = gr.File(
            label="Load Text to Generate from File (.txt)",
            file_types=[".txt"],
            scale=1,
        )

    with gr.Row():
        clear_btn = gr.Button("Clear")
        paste_btn = gr.Button("Paste")
        copy_btn = gr.Button("Copy")
        ruaccent_btn = gr.Button("RUAccent")

    clear_btn.click(lambda: "", inputs=None, outputs=gen_text_input)

    paste_btn.click(
        None,
        inputs=None,
        outputs=gen_text_input,
        js="""
        async () => {
            try {
                const text = await navigator.clipboard.readText();
                return text;
            } catch (err) {
                alert("Error accessing clipboard: " + err);
                return "";
            }
        }
        """
    )

    copy_btn.click(
        None,
        inputs=[gen_text_input],
        outputs=None,
        js="(text) => navigator.clipboard.writeText(text)"
    )

    ruaccent_btn.click(
        preprocess_text,
        inputs=[gen_text_input],
        outputs=gen_text_input,
    )

    with gr.Row():
        generate_btn = gr.Button("Generate", variant="primary")
        restart_btn = gr.Button("Restart Engine", variant="stop")

    def restart_engine():
        import sys, os, threading, time
        def _delayed_restart():
            time.sleep(2)
            python = sys.executable
            os.execl(python, python, *sys.argv)
        threading.Thread(target=_delayed_restart).start()
        return "⚠ Restarting engine..."

    restart_btn.click(
        restart_engine,
        inputs=None,
        outputs=None,
        js="() => { setTimeout(() => { window.location.reload(); }, 30000); }"
    )

    with gr.Accordion("Advanced Settings", open=True) as adv_settn:
        with gr.Row():
            randomize_seed = gr.Checkbox(
                label="Randomize Seed",
                info="Check to use a random seed for each generation. Uncheck to use the seed specified.",
                value=True,
                scale=3,
            )
            seed_input = gr.Number(show_label=False, value=0, precision=0, scale=1)
            with gr.Column(scale=4):
                remove_silence = gr.Checkbox(
                    label="Remove Silences",
                    info="If undesired long silence(s) produced, turn on to automatically detect and crop.",
                    value=False,
                )
        speed_slider = gr.Slider(
            label="Speed",
            minimum=0.3,
            maximum=2.0,
            value=1.0,
            step=0.1,
            info="Adjust the speed of the audio.",
        )
        nfe_slider = gr.Slider(
            label="NFE Steps",
            minimum=4,
            maximum=64,
            value=16,
            step=2,
            info="Set the number of denoising steps. Less steps - faster, but worst quality.",
        )
        cross_fade_duration_slider = gr.Slider(
            label="Cross-Fade Duration (seconds)",
            minimum=0.0,
            maximum=1.0,
            value=0.0,
            step=0.01,
            info="Set the duration of the cross-fade between audio clips.",
        )
        target_rms_input = gr.Slider(
            label="Target RMS",
            minimum=0.01,
            maximum=1.0,
            value=0.1,
            step=0.01,
            info="Target volume level (RMS)",
        )
        cfg_strength_input = gr.Slider(
            label="CFG Strength",
            minimum=0.1,
            maximum=5.0,
            value=1.0,
            step=0.1,
            info="Classifier-Free Guidance",
        )
        use_accent_checkbox = gr.Checkbox(
            label="RuAccent auto pronounce.",
            info="Automatic RuAccent pre-processing for russian language pronounce.",
            value=False,
        )


    def collapse_accordion():
        return gr.Accordion(open=True)

    # Workaround for https://github.com/SWivid/F5-TTS/issues/1239#issuecomment-3677987413
    # i.e. to set gr.Accordion(open=True) by default, then collapse manually Blocks loaded
    app_tts.load(
        fn=collapse_accordion,
        inputs=None,
        outputs=adv_settn,
    )

    audio_output = gr.Audio(label="Generated Audio")
    saved_output = gr.File(label="Saved Output")
    spectrogram_output = gr.Image(label="Spectrogram")

    @gpu_decorator
    def basic_tts(
        ref_audio_input,
        ref_text_input,
        gen_text_input,
        remove_silence,
        randomize_seed,
        seed_input,
        cross_fade_duration_slider,
        nfe_slider,
        speed_slider,
        use_accent,
        target_rms_input,
        cfg_strength_input,
        output_format,
    ):
        if randomize_seed:
            seed_input = np.random.randint(0, 2**31 - 1)

        audio_out, spectrogram_path, ref_text_out, used_seed = infer(
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            tts_model_choice,
            remove_silence,
            seed=seed_input,
            cross_fade_duration=cross_fade_duration_slider,
            nfe_step=nfe_slider,
            speed=speed_slider,
            use_accent=use_accent,
            target_rms=target_rms_input,
            cfg_strength=cfg_strength_input,
        )
        
        out_path = save_audio(audio_out[1], audio_out[0], output_format, used_seed)
        
        return audio_out, spectrogram_path, ref_text_out, used_seed, out_path

    gen_text_file.upload(
        load_text_from_file,
        inputs=[gen_text_file],
        outputs=[gen_text_input],
    )

    ref_text_file.upload(
        load_text_from_file,
        inputs=[ref_text_file],
        outputs=[ref_text_input],
    )

    ref_audio_input.clear(
        lambda: [None, None],
        None,
        [ref_text_input, ref_text_file],
    )

    generate_btn.click(
        basic_tts,
        inputs=[
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            remove_silence,
            randomize_seed,
            seed_input,
            cross_fade_duration_slider,
            nfe_slider,
            speed_slider,
            use_accent_checkbox,
            target_rms_input,
            cfg_strength_input,
            format_selector,
        ],
        outputs=[audio_output, spectrogram_output, ref_text_input, seed_input, saved_output,],
    )


def parse_speechtypes_text(gen_text):
    # Pattern to find {str} or {"name": str, "seed": int, "speed": float}
    pattern = r"(\{.*?\})"

    # Split the text by the pattern
    tokens = re.split(pattern, gen_text)

    segments = []

    current_type_dict = {
        "name": "Regular",
        "seed": -1,
        "speed": 1.0,
    }

    for i in range(len(tokens)):
        if i % 2 == 0:
            # This is text
            text = tokens[i].strip()
            if text:
                current_type_dict["text"] = text
                segments.append(current_type_dict)
        else:
            # This is type
            type_str = tokens[i].strip()
            try:  # if type dict
                current_type_dict = json.loads(type_str)
            except json.decoder.JSONDecodeError:
                type_str = type_str[1:-1]  # remove brace {}
                current_type_dict = {"name": type_str, "seed": -1, "speed": 1.0}

    return segments


with gr.Blocks() as app_multistyle:
    # New section for multistyle generation
    gr.Markdown(
        """
    # Multiple Speech-Type Generation

    This section allows you to generate multiple speech types or multiple people's voices. Enter your text in the format shown below, or upload a .txt file with the same format. The system will generate speech using the appropriate type. If unspecified, the model will use the regular speech type. The current speech type will be used until the next speech type is specified.
    """
    )

    with gr.Row():
        gr.Markdown(
            """
            **Example Input:** <br>
            {Regular} Hello, I'd like to order a sandwich please. <br>
            {Surprised} What do you mean you're out of bread? <br>
            {Sad} I really wanted a sandwich though... <br>
            {Angry} You know what, darn you and your little shop! <br>
            {Whisper} I'll just go back home and cry now. <br>
            {Shouting} Why me?!
            """
        )

        gr.Markdown(
            """
            **Example Input 2:** <br>
            {"name": "Speaker1_Happy", "seed": -1, "speed": 1} Hello, I'd like to order a sandwich please. <br>
            {"name": "Speaker2_Regular", "seed": -1, "speed": 1} Sorry, we're out of bread. <br>
            {"name": "Speaker1_Sad", "seed": -1, "speed": 1} I really wanted a sandwich though... <br>
            {"name": "Speaker2_Whisper", "seed": -1, "speed": 1} I'll give you the last one I was hiding.
            """
        )

    gr.Markdown(
        'Upload different audio clips for each speech type. The first speech type is mandatory. You can add additional speech types by clicking the "Add Speech Type" button.'
    )

    # Regular speech type (mandatory)
    with gr.Row(variant="compact") as regular_row:
        # Left narrow column
        with gr.Column(scale=1, min_width=160):
            regular_name = gr.Textbox(value="Regular", label="Speech Type Name")
            regular_insert = gr.Button("Insert Label", variant="secondary")

        # Center speaker: voice + audio
        with gr.Column(scale=3):
            voice_selector = gr.Dropdown(
                choices=voice_choices,
                label="Pre-defined voices (folder 'voices')",
                value="-NONE-",
            )
            # Update btn
            refresh_btn = gr.Button("🔄", elem_classes="square-btn")
            
            regular_audio = gr.Audio(label="Regular Reference Audio", type="filepath")

        # Right column: text field + sliders
        with gr.Column(scale=3):
            regular_ref_text = gr.Textbox(label="Reference Text (Regular)", lines=4)

            with gr.Column():
                regular_seed_slider = gr.Slider(
                    show_label=False, minimum=-1, maximum=999, value=-1, step=1, info="Seed, -1 for random"
                )
                regular_speed_slider = gr.Slider(
                    label="Speed",
                    minimum=0.3, maximum=2.0, value=1.0, step=0.1
                )
                regular_crossfade_slider = gr.Slider(
                    label="Cross-Fade Duration (seconds)",
                    minimum=0.0, maximum=1.0, value=0.15, step=0.01
                )
                regular_nfe_slider = gr.Slider(
                    label="NFE Steps",
                    minimum=4, maximum=64, value=16, step=2
                )
                regular_cfg_slider = gr.Slider(
                    label="CFG Strength",
                    minimum=0.1, maximum=5.0, value=2.0, step=0.1
                )
                regular_accent_checkbox = gr.Checkbox(
                    label="RuAccent",
                    value=False,
                )


        # Right narrow column for loading text from a file
        with gr.Column(scale=1, min_width=160):
            regular_ref_text_file = gr.File(
                label="Load Reference Text from File (.txt)",
                file_types=[".txt"]
            )

    refresh_btn.click(
        lambda: gr.update(choices=get_voice_choices(), value="-NONE-"),
        inputs=None,
        outputs=voice_selector,
    )

    voice_selector.change(
        lambda: (None, ""),
        inputs=None,
        outputs=[regular_audio, regular_ref_text],
    ).then(
        set_voice_file,
        inputs=[voice_selector],
        outputs=[regular_audio],
    )

    # Regular speech type (max 100)
    max_speech_types = 100
    speech_type_rows = [regular_row]
    speech_type_names = [regular_name]
    speech_type_audios = [regular_audio]
    speech_type_ref_texts = [regular_ref_text]
    speech_type_ref_text_files = [regular_ref_text_file]
    speech_type_seeds = [regular_seed_slider]
    speech_type_speeds = [regular_speed_slider]
    speech_type_delete_btns = [None]
    speech_type_insert_btns = [regular_insert]
    speech_type_crossfades = [regular_crossfade_slider]
    speech_type_nfes = [regular_nfe_slider]
    speech_type_cfgs = [regular_cfg_slider]
    speech_type_accent_checkboxes = [regular_accent_checkbox]

    # Additional speech types (99 more)
    for i in range(max_speech_types - 1):
        with gr.Row(variant="compact", visible=False) as row:
            # Left narrow column
            with gr.Column(scale=1, min_width=160):
                name_input = gr.Textbox(label="Speech Type Name")
                insert_btn = gr.Button("Insert Label", variant="secondary")
                delete_btn = gr.Button("Delete Type", variant="stop")

            # Center speaker: voice + audio
            with gr.Column(scale=3):
                voice_selector = gr.Dropdown(
                    choices=voice_choices,
                    label="Pre-defined voices (folder 'voices')",
                    value="-NONE-",
                )
                # Update btn
                refresh_btn = gr.Button("🔄", elem_classes="square-btn")
                
                audio_input = gr.Audio(label="Reference Audio", type="filepath")

            # Right column: text field + sliders
            with gr.Column(scale=3):
                ref_text_input = gr.Textbox(label="Reference Text", lines=4)

                with gr.Column():
                    seed_input = gr.Slider(
                        show_label=False, minimum=-1, maximum=999, value=-1, step=1,
                        info="Seed. -1 for random"
                    )
                    speed_input = gr.Slider(
                        label="Speed",
                        minimum=0.3, maximum=2.0, value=1.0, step=0.1
                    )
                    crossfade_input = gr.Slider(
                        label="Cross-Fade Duration (seconds)",
                        minimum=0.0, maximum=1.0, value=0.15, step=0.01
                    )
                    nfe_input = gr.Slider(
                        label="NFE Steps",
                        minimum=4, maximum=64, value=16, step=2
                    )
                    cfg_input = gr.Slider(
                        label="CFG Strength",
                        minimum=0.1, maximum=5.0, value=2.0, step=0.1
                    )
                    regular_accent_checkbox = gr.Checkbox(
                        label="RuAccent",
                        value=False,
                    )


            # Right narrow column for loading text from a file
            with gr.Column(scale=1, min_width=160):
                ref_text_file_input = gr.File(
                    label="Load Reference Text from File (.txt)",
                    file_types=[".txt"]
                )

        refresh_btn.click(
            lambda: gr.update(choices=get_voice_choices(), value="-NONE-"),
            inputs=None,
            outputs=voice_selector,
        )

        # File clearing and installation logic - after all components are declared
        voice_selector.change(
            lambda: (None, ""),
            inputs=None,
            outputs=[audio_input, ref_text_input],
        ).then(
            set_voice_file,
            inputs=[voice_selector],
            outputs=[audio_input],
        )

        speech_type_rows.append(row)
        speech_type_names.append(name_input)
        speech_type_audios.append(audio_input)
        speech_type_ref_texts.append(ref_text_input)
        speech_type_ref_text_files.append(ref_text_file_input)
        speech_type_seeds.append(seed_input)
        speech_type_speeds.append(speed_input)
        speech_type_delete_btns.append(delete_btn)
        speech_type_insert_btns.append(insert_btn)
        speech_type_crossfades.append(crossfade_input)
        speech_type_nfes.append(nfe_input)
        speech_type_cfgs.append(cfg_input)
        speech_type_accent_checkboxes.append(regular_accent_checkbox)


    # Global logic for all speech types
    for i in range(max_speech_types):
        speech_type_audios[i].clear(
            lambda: [None, None],
            None,
            [speech_type_ref_texts[i], speech_type_ref_text_files[i]],
        )
        speech_type_ref_text_files[i].upload(
            load_text_from_file,
            inputs=[speech_type_ref_text_files[i]],
            outputs=[speech_type_ref_texts[i]],
        )

    # Button to add speech type
    add_speech_type_btn = gr.Button("Add Speech Type")

    # Keep track of autoincrement of speech types, no roll back
    speech_type_count = 1

    # Function to add a speech type
    def add_speech_type_fn():
        row_updates = [gr.update() for _ in range(max_speech_types)]
        global speech_type_count
        if speech_type_count < max_speech_types:
            row_updates[speech_type_count] = gr.update(visible=True)
            speech_type_count += 1
        else:
            gr.Warning("Exhausted maximum number of speech types. Consider restart the app.")
        return row_updates

    add_speech_type_btn.click(add_speech_type_fn, outputs=speech_type_rows)

    # Function to delete a speech type
    def delete_speech_type_fn():
        return gr.update(visible=False), None, None, None, None

    # Update delete button clicks and ref text file changes
    for i in range(1, len(speech_type_delete_btns)):
        speech_type_delete_btns[i].click(
            delete_speech_type_fn,
            outputs=[
                speech_type_rows[i],
                speech_type_names[i],
                speech_type_audios[i],
                speech_type_ref_texts[i],
                speech_type_ref_text_files[i],
            ],
        )

    # Text input for the prompt
    with gr.Row():
        gen_text_input_multistyle = gr.Textbox(
            label="Text to Generate",
            lines=10,
            max_lines=40,
            scale=4,
            placeholder="Enter the script with speaker names (or emotion types) at the start of each block, e.g.:\n\n{Regular} Hello, I'd like to order a sandwich please.\n{Surprised} What do you mean you're out of bread?\n{Sad} I really wanted a sandwich though...\n{Angry} You know what, darn you and your little shop!\n{Whisper} I'll just go back home and cry now.\n{Shouting} Why me?!",
        )
        gen_text_file_multistyle = gr.File(label="Load Text to Generate from File (.txt)", file_types=[".txt"], scale=1)

    with gr.Row():
        clear_btn = gr.Button("Clear")
        paste_btn = gr.Button("Paste")
        copy_btn = gr.Button("Copy")
        ruaccent_btn = gr.Button("RUAccent")

    clear_btn.click(
        lambda: "",
        inputs=None,
        outputs=gen_text_input_multistyle,
    )

    paste_btn.click(
        None,
        inputs=None,
        outputs=gen_text_input_multistyle,
        js="""
        async () => {
            try {
                const text = await navigator.clipboard.readText();
                return text;
            } catch (err) {
                alert("Error accessing clipboard: " + err);
                return "";
            }
        }
        """
    )

    copy_btn.click(
        None,
        inputs=[gen_text_input_multistyle],
        outputs=None,
        js="(text) => navigator.clipboard.writeText(text)"
    )

    ruaccent_btn.click(
        preprocess_text,
        inputs=[gen_text_input_multistyle],
        outputs=gen_text_input_multistyle,
    )


    def make_insert_speech_type_fn(index):
        def insert_speech_type_fn(current_text, speech_type_name, speech_type_seed, speech_type_speed):
            current_text = current_text or ""
            if not speech_type_name:
                gr.Warning("Please enter speech type name before insert.")
                return current_text
            speech_type_dict = {
                "name": speech_type_name,
                "seed": speech_type_seed,
                "speed": speech_type_speed,
            }
            updated_text = current_text + json.dumps(speech_type_dict) + " "
            return updated_text

        return insert_speech_type_fn

    for i, insert_btn in enumerate(speech_type_insert_btns):
        insert_fn = make_insert_speech_type_fn(i)
        insert_btn.click(
            insert_fn,
            inputs=[gen_text_input_multistyle, speech_type_names[i], speech_type_seeds[i], speech_type_speeds[i]],
            outputs=gen_text_input_multistyle,
        )

    with gr.Accordion("Advanced Settings", open=True):
        with gr.Row():
            with gr.Column():
                show_cherrypick_multistyle = gr.Checkbox(
                    label="Show Cherry-pick Interface",
                    info="Turn on to show interface, picking seeds from previous generations.",
                    value=False,
                )
            with gr.Column():
                remove_silence_multistyle = gr.Checkbox(
                    label="Remove Silences",
                    info="Turn on to automatically detect and crop long silences.",
                    value=False,
                )
            with gr.Row():
                format_selector = gr.Dropdown(
                    choices=["wav", "mp3", "aac", "m4a", "m4b", "ogg", "flac", "opus"],
                    value="wav",
                    label="Output Format",
                )

    # Generate button
    with gr.Row():
        generate_multistyle_btn = gr.Button("Generate Multi-Style Speech", variant="primary")
        restart_btn = gr.Button("Restart Engine", variant="stop")

    def restart_engine():
        import sys, os, threading, time
        def _delayed_restart():
            time.sleep(2)
            python = sys.executable
            os.execl(python, python, *sys.argv)
        threading.Thread(target=_delayed_restart).start()
        return "⚠ Restarting engine..."

    restart_btn.click(
        restart_engine,
        inputs=None,
        outputs=None,
        js="() => { setTimeout(() => { window.location.reload(); }, 30000); }"
    )

    # Output audio
    audio_output_multistyle = gr.Audio(label="Generated Audio")

    # Saved output file
    saved_output = gr.File(label="Saved Output")

    # Used seed gallery
    cherrypick_interface_multistyle = gr.Textbox(
        label="Cherry-pick Interface",
        lines=10,
        max_lines=40,
        buttons=["copy"],  # show_copy_button=True if gradio<6.0
        interactive=False,
        visible=False,
    )

    # Logic control to show/hide the cherrypick interface
    show_cherrypick_multistyle.change(
        lambda is_visible: gr.update(visible=is_visible),
        show_cherrypick_multistyle,
        cherrypick_interface_multistyle,
    )

    # Function to load text to generate from file
    gen_text_file_multistyle.upload(
        load_text_from_file,
        inputs=[gen_text_file_multistyle],
        outputs=[gen_text_input_multistyle],
    )

    @gpu_decorator
    def generate_multistyle_speech(gen_text, *args):
        # the last argument is the selected format
        *args, output_format = args

        # Parsing arguments into blocks
        speech_type_names_list     = args[:max_speech_types]
        speech_type_audios_list    = args[max_speech_types : 2 * max_speech_types]
        speech_type_ref_texts_list = args[2 * max_speech_types : 3 * max_speech_types]

        remove_silence = args[3 * max_speech_types]

        speech_type_crossfades_list = args[3 * max_speech_types + 1 : 4 * max_speech_types + 1]
        speech_type_nfes_list       = args[4 * max_speech_types + 1 : 5 * max_speech_types + 1]
        speech_type_cfgs_list       = args[5 * max_speech_types + 1 : 6 * max_speech_types + 1]
        speech_type_accent_list     = args[6 * max_speech_types + 1 : 7 * max_speech_types + 1]

        # Building the speech_types dictionary
        speech_types = OrderedDict()
        ref_text_idx = 0
        for name_input, audio_input, ref_text_input in zip(
            speech_type_names_list, speech_type_audios_list, speech_type_ref_texts_list
        ):
            if name_input and audio_input:
                speech_types[name_input] = {"audio": audio_input, "ref_text": ref_text_input}
            else:
                speech_types[f"@{ref_text_idx}@"] = {"audio": "", "ref_text": ""}
            ref_text_idx += 1

        # Parsing text
        segments = parse_speechtypes_text(gen_text)

        generated_audio_segments = []
        current_type_name = "Regular"
        inference_meta_data = ""

        for i, segment in enumerate(segments):
            name = segment["name"]
            seed_input = segment["seed"]
            speed = segment["speed"]
            text = segment["text"]

            if name in speech_types:
                current_type_name = name
            else:
                gr.Warning(f"Type {name} is not available, will use Regular as default.")
                current_type_name = "Regular"

            try:
                ref_audio = speech_types[current_type_name]["audio"]
            except KeyError:
                gr.Warning(f"Please provide reference audio for type {current_type_name}.")
                return [None] + [speech_types[name]["ref_text"] for name in speech_types] + [None]

            ref_text = speech_types[current_type_name].get("ref_text", "")

            if seed_input == -1:
                seed_input = np.random.randint(0, 2**31 - 1)

            # If the checkbox is enabled, run the text through RuAccent
            if speech_type_accent_list[i]:
                text = preprocess_text(text)

            # call infer
            audio_out, _, ref_text_out, used_seed = infer(
                ref_audio,
                ref_text,
                text,
                tts_model_choice,
                remove_silence,
                seed=seed_input,
                cross_fade_duration=speech_type_crossfades_list[i],
                nfe_step=speech_type_nfes_list[i],
                speed=speed,
                cfg_strength=speech_type_cfgs_list[i],
                show_info=print,
            )
            sr, audio_data = audio_out

            generated_audio_segments.append(audio_data)
            speech_types[current_type_name]["ref_text"] = ref_text_out
            inference_meta_data += json.dumps(dict(name=name, seed=used_seed, speed=speed)) + f" {text}\n"

        # Glue the segments and save a single file
        if generated_audio_segments:
            final_audio_data = np.concatenate(generated_audio_segments)

            # save only the resulting file
            out_path = save_audio(final_audio_data, sr, output_format, seed_input)

            return (
                [(sr, final_audio_data)]
                + [speech_types[name]["ref_text"] for name in speech_types]
                + [inference_meta_data]
                + [out_path]  # path to a single file
            )
        else:
            gr.Warning("No audio generated.")
            return [None] + [speech_types[name]["ref_text"] for name in speech_types] + [None]


    # generate multistyle btn
    generate_multistyle_btn.click(
        generate_multistyle_speech,
        inputs=[
            gen_text_input_multistyle,
        ]
        + speech_type_names
        + speech_type_audios
        + speech_type_ref_texts
        + [remove_silence_multistyle]
        + speech_type_crossfades
        + speech_type_nfes
        + speech_type_cfgs
        + speech_type_accent_checkboxes
        + [format_selector],
        outputs=[
            audio_output_multistyle
        ]
        + speech_type_ref_texts
        + [cherrypick_interface_multistyle, saved_output],
    )


    # Validation function to disable Generate button if speech types are missing
    def validate_speech_types(gen_text, regular_name, *args):
        speech_type_names_list = args

        # Collect the speech types names
        speech_types_available = set()
        if regular_name:
            speech_types_available.add(regular_name)
        for name_input in speech_type_names_list:
            if name_input:
                speech_types_available.add(name_input)

        # Parse the gen_text to get the speech types used
        segments = parse_speechtypes_text(gen_text)
        speech_types_in_text = set(segment["name"] for segment in segments)

        # Check if all speech types in text are available
        missing_speech_types = speech_types_in_text - speech_types_available

        if missing_speech_types:
            # Disable the generate button
            return gr.update(interactive=False)
        else:
            # Enable the generate button
            return gr.update(interactive=True)

    gen_text_input_multistyle.change(
        validate_speech_types,
        inputs=[gen_text_input_multistyle, regular_name] + speech_type_names,
        outputs=generate_multistyle_btn,
    )


with gr.Blocks() as app_chat:
    gr.Markdown(
        """
# Voice Chat
Have a conversation with an AI using your reference voice!
1. Upload a reference audio clip and optionally its transcript (via text or .txt file).
2. Load the chat model.
3. Record your message through your microphone or type it.
4. The AI will respond using the reference voice.
"""
    )

    chat_model_name_list = [
        "Qwen/Qwen2.5-3B-Instruct",
        "microsoft/Phi-4-mini-instruct",
    ]

    @gpu_decorator
    def load_chat_model(chat_model_name):
        show_info = gr.Info
        global chat_model_state, chat_tokenizer_state
        if chat_model_state is not None:
            chat_model_state = None
            chat_tokenizer_state = None
            gc.collect()
            torch.cuda.empty_cache()

        show_info(f"Loading chat model: {chat_model_name}")
        chat_model_state = AutoModelForCausalLM.from_pretrained(chat_model_name, torch_dtype="auto", device_map="auto")
        chat_tokenizer_state = AutoTokenizer.from_pretrained(chat_model_name)
        show_info(f"Chat model {chat_model_name} loaded successfully!")

        return gr.update(visible=False), gr.update(visible=True)

    if USING_SPACES:
        load_chat_model(chat_model_name_list[0])

    chat_model_name_input = gr.Dropdown(
        choices=chat_model_name_list,
        value=chat_model_name_list[0],
        label="Chat Model Name",
        info="Enter the name of a HuggingFace chat model",
        allow_custom_value=not USING_SPACES,
    )
    load_chat_model_btn = gr.Button("Load Chat Model", variant="primary", visible=not USING_SPACES)
    chat_interface_container = gr.Column(visible=USING_SPACES)

    chat_model_name_input.change(
        lambda: gr.update(visible=True),
        None,
        load_chat_model_btn,
        show_progress="hidden",
    )
    load_chat_model_btn.click(
        load_chat_model, inputs=[chat_model_name_input], outputs=[load_chat_model_btn, chat_interface_container]
    )

    with chat_interface_container:
        with gr.Row():
            with gr.Column():
                voice_selector = gr.Dropdown(
                    choices=voice_choices,
                    label="Pre-defined voices (folder 'voices')",
                    value="-NONE-",
                )
                # Update btn
                refresh_btn = gr.Button("🔄", elem_classes="square-btn")

                ref_audio_chat = gr.Audio(label="Reference Audio", type="filepath")
            with gr.Column():
                with gr.Accordion("Advanced Settings", open=False):
                    with gr.Row():
                        ref_text_chat = gr.Textbox(
                            label="Reference Text",
                            info="Optional: Leave blank to auto-transcribe",
                            lines=2,
                            scale=3,
                        )
                        ref_text_file_chat = gr.File(
                            label="Load Reference Text from File (.txt)", file_types=[".txt"], scale=1
                        )
                    with gr.Row():
                        randomize_seed_chat = gr.Checkbox(
                            label="Randomize Seed",
                            value=True,
                            info="Uncheck to use the seed specified.",
                            scale=3,
                        )
                        seed_input_chat = gr.Number(show_label=False, value=0, precision=0, scale=1)
                    remove_silence_chat = gr.Checkbox(
                        label="Remove Silences",
                        value=False,
                    )

                    with gr.Column():
                        speed_slider_chat = gr.Slider(
                            label="Speed",
                            minimum=0.3,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            info="Adjust the speed of the audio.",
                        )
                        nfe_slider_chat = gr.Slider(
                            label="NFE Steps",
                            minimum=4,
                            maximum=64,
                            value=16,
                            step=2,
                            info="Set the number of denoising steps. Less steps - faster, but worse quality.",
                        )
                        crossfade_slider_chat = gr.Slider(
                            label="Cross-Fade Duration (seconds)",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.15,
                            step=0.01,
                            info="Set the duration of the cross-fade between audio clips.",
                        )
                        cfg_strength_chat = gr.Slider(
                            label="CFG Strength",
                            minimum=0.1,
                            maximum=5.0,
                            value=2.0,
                            step=0.1,
                            info="Classifier-Free Guidance strength.",
                        )
                        use_accent_chat = gr.Checkbox(
                            label="RuAccent",
                            info="Automatic RuAccent pre-processing for Russian language pronounce.",
                            value=False,
                        )

                    system_prompt_chat = gr.Textbox(
                        label="System Prompt",
                        value="You are not an AI assistant, you are whoever the user says you are. You must stay in character. Keep your responses concise since they will be spoken out loud.",
                        lines=2,
                    )
            refresh_btn.click(
                lambda: gr.update(choices=get_voice_choices(), value="-NONE-"),
                inputs=None,
                outputs=voice_selector,
            )

            # File clearing and installation logic - after all components are declared
            voice_selector.change(
                lambda: (None, ""),
                inputs=None,
                outputs=[ref_audio_chat, ref_text_chat],
            ).then(
                set_voice_file,
                inputs=[voice_selector],
                outputs=[ref_audio_chat],
            )

        chatbot_interface = gr.Chatbot(
            label="Conversation"
        )  # type="messages" hard-coded and no need to pass in since gradio 6.0

        with gr.Row():
            with gr.Column():
                audio_input_chat = gr.Microphone(
                    label="Speak your message",
                    type="filepath",
                )
                audio_output_chat = gr.Audio(autoplay=True)
            with gr.Column():
                text_input_chat = gr.Textbox(
                    label="Type your message",
                    lines=1,
                )
                send_btn_chat = gr.Button("Send Message")
                clear_btn_chat = gr.Button("Clear Conversation")

        # Modify process_audio_input to generate user input
        @gpu_decorator
        def process_audio_input(conv_state, audio_path, text):
            """Handle audio or text input from user"""

            if not audio_path and not text.strip():
                return conv_state

            if audio_path:
                text = preprocess_ref_audio_text(audio_path, text)[1]
            if not text.strip():
                return conv_state

            conv_state.append({"role": "user", "content": text})
            return conv_state

        # Use model and tokenizer from state to get text response
        @gpu_decorator
        def generate_text_response(conv_state, system_prompt):
            """Generate text response from AI"""
            for single_state in conv_state:
                if isinstance(single_state["content"], list):
                    assert len(single_state["content"]) == 1 and single_state["content"][0]["type"] == "text"
                    single_state["content"] = single_state["content"][0]["text"]

            system_prompt_state = [{"role": "system", "content": system_prompt}]
            response = chat_model_inference(system_prompt_state + conv_state, chat_model_state, chat_tokenizer_state)

            conv_state.append({"role": "assistant", "content": response})
            return conv_state

        @gpu_decorator
        def generate_audio_response(
            conv_state,
            ref_audio,
            ref_text,
            remove_silence,
            randomize_seed,
            seed_input,
            speed,
            crossfade,
            cfg_strength,
            nfe_steps,
            use_accent,
        ):
            """Generate TTS audio for AI response"""
            if not conv_state or not ref_audio:
                return None, ref_text, seed_input

            last_ai_response = conv_state[-1]["content"][0]["text"]
            if not last_ai_response or conv_state[-1]["role"] != "assistant":
                return None, ref_text, seed_input

            if randomize_seed:
                seed_input = np.random.randint(0, 2**31 - 1)

            audio_result, _, ref_text_out, used_seed = infer(
                ref_audio,
                ref_text,
                last_ai_response,
                tts_model_choice,
                remove_silence,
                seed=seed_input,
                cross_fade_duration=crossfade,
                nfe_step=nfe_steps,
                speed=speed,
                use_accent=use_accent,
                cfg_strength=cfg_strength,
            )
            return audio_result, ref_text_out, used_seed


        def clear_conversation():
            """Reset the conversation"""
            return [], None

        ref_text_file_chat.upload(
            load_text_from_file,
            inputs=[ref_text_file_chat],
            outputs=[ref_text_chat],
        )

        for user_operation in [audio_input_chat.stop_recording, text_input_chat.submit, send_btn_chat.click]:
            user_operation(
                process_audio_input,
                inputs=[chatbot_interface, audio_input_chat, text_input_chat],
                outputs=[chatbot_interface],
            ).then(
                generate_text_response,
                inputs=[chatbot_interface, system_prompt_chat],
                outputs=[chatbot_interface],
            ).then(
                generate_audio_response,
                inputs=[
                    chatbot_interface,
                    ref_audio_chat,
                    ref_text_chat,
                    remove_silence_chat,
                    randomize_seed_chat,
                    seed_input_chat,
                    speed_slider_chat,
                    crossfade_slider_chat,
                    cfg_strength_chat,
                    nfe_slider_chat,
                    use_accent_chat,
                ],
                outputs=[audio_output_chat, ref_text_chat, seed_input_chat],
            ).then(
                lambda: [None, None],
                None,
                [audio_input_chat, text_input_chat],
            )

        # Handle clear button or system prompt change and reset conversation
        for user_operation in [clear_btn_chat.click, system_prompt_chat.change, chatbot_interface.clear]:
            user_operation(
                clear_conversation,
                outputs=[chatbot_interface, audio_output_chat],
            )


with gr.Blocks() as app_additional:
    gr.Markdown("""
**Additional information**

F5-TTSx based on [F5-TTS](https://github.com/SWivid/F5-TTS) sources.

F5-TTXx official [Git](https://github.com/LeeAeron/F5-TTSx)

**Supported Languages**

- [Multilingual (zh & en)](https://huggingface.co/SWivid/F5-TTS/tree/main/F5TTS_v1_Base)
- [English](#english)
- [Finnish](https://huggingface.co/AsmoKoskinen/F5-TTS_Finnish_Model)
- [French](https://huggingface.co/RASPIAUDIO/F5-French-MixedSpeakers-reduced)
- [German](https://huggingface.co/hvoss-techfak/F5-TTS-German)
- [Hindi](https://huggingface.co/SPRINGLab/F5-Hindi-24KHz)
- [Italian](https://huggingface.co/alien79/F5-TTS-italian)
- [Japanese](https://huggingface.co/Jmica/F5TTS/tree/main/JA_21999120)
- [Latvian](https://huggingface.co/RaivisDejus/F5-TTS-Latvian)
- Mandarin
- [Russian](https://huggingface.co/hotstone228/F5-TTS-Russian)
- [Russian](https://huggingface.co/Misha24-10/F5-TTS_RUSSIAN)
- [Spanish](https://huggingface.co/jpgallegoar/F5-Spanish)

**Credits**

* [LeeAeron](https://github.com/LeeAeron) — additional code, repository, Hugginface space, features, installer/launcher, reference audios, dictionary support.
* [mrfakename](https://github.com/fakerybakery) — original [online demo](https://huggingface.co/spaces/mrfakename/E2-F5-TTS)
* [RootingInLoad](https://github.com/RootingInLoad) — chunk generation & podcast app exploration
* [jpgallegoar](https://github.com/jpgallegoar) — multiple speech-type generation & voice chat
* [Ebany Speech](https://huggingface.co/ESpeech) - additional russian language TTS models
"""
)


with gr.Blocks(title="F5-TTSx") as app:
    gr.Markdown(
f"""
# F5-TTSx
"""
    )
    last_used_custom = files("f5_tts").joinpath("infer/.cache/last_used_custom_model_info_v1.txt")
    
    # always default at startup
    tts_model_choice = DEFAULT_TTS_MODEL  

    def load_last_used_custom():
        try:
            custom = []
            with open(last_used_custom, "r", encoding="utf-8") as f:
                for line in f:
                    custom.append(line.strip())
            return custom
        except FileNotFoundError:
            last_used_custom.parent.mkdir(parents=True, exist_ok=True)
            return DEFAULT_TTS_MODEL_CFG

    def switch_tts_model(new_choice):
        global tts_model_choice
        if new_choice == "Custom":
            custom_ckpt_path, custom_vocab_path, custom_model_cfg = load_last_used_custom()
            tts_model_choice = ("Custom", custom_ckpt_path, custom_vocab_path, custom_model_cfg)
            return (
                gr.update(visible=True, value=custom_ckpt_path),
                gr.update(visible=True, value=custom_vocab_path),
                gr.update(visible=True, value=custom_model_cfg),
                gr.update()
            )

        elif new_choice == "Misha24-10_v2":
            ckpt = safe_path(
                "models/F5TTS_RU/v2/Misha24-10_v2.safetensors",
                "https://huggingface.co/datasets/LeeAeron/F5TTSx/resolve/main/models/F5TTS_RU/v2/Misha24-10_v2.safetensors"
            )
            vocab = safe_path(
                "models/F5TTS_RU/v2/vocab.txt",
                "https://huggingface.co/datasets/LeeAeron/F5TTSx/resolve/main/models/F5TTS_RU/v2/vocab.txt"
            )
            tts_model_choice = ("Misha24-10_v2", ckpt, vocab,
                                json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2,
                                                text_dim=512, conv_layers=4)))
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=True)

        elif new_choice == "Misha24-10_v4":
            ckpt = safe_path(
                "models/F5TTS_RU/v4_winter/Misha24-10_v4_winter.safetensors",
                "https://huggingface.co/datasets/LeeAeron/F5TTSx/resolve/main/models/F5TTS_RU/v4_winter/Misha24-10_v4_winter.safetensors"
            )
            vocab = safe_path(
                "models/F5TTS_RU/v4_winter/vocab.txt",
                "https://huggingface.co/datasets/LeeAeron/F5TTSx/resolve/main/models/F5TTS_RU/v4_winter/vocab.txt"
            )
            tts_model_choice = ("Misha24-10_v4", ckpt, vocab,
                                json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2,
                                                text_dim=512, conv_layers=4)))
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=True)

        elif new_choice == "ESpeech-TTS-1_podcaster":
            ckpt = safe_path(
                "models/ESpeech/ESpeech-TTS-1_podcaster/espeech_tts_podcaster.pt",
                "https://huggingface.co/datasets/LeeAeron/F5TTSx/resolve/main/models/ESpeech-TTS-1_podcaster/espeech_tts_podcaster.pt"
            )
            vocab = safe_path(
                "models/ESpeech/ESpeech-TTS-1_podcaster/vocab.txt",
                "https://huggingface.co/datasets/LeeAeron/F5TTSx/resolve/main/models/ESpeech-TTS-1_podcaster/vocab.txt"
            )
            tts_model_choice = ("ESpeech-TTS-1_podcaster", ckpt, vocab,
                                json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2,
                                                text_dim=512, conv_layers=4)))
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=True)

        elif new_choice == "ESpeech-TTS-1_RL-V1":
            ckpt = safe_path(
                "models/ESpeech/ESpeech-TTS-1_RL-V1/espeech_tts_rlv1.pt",
                "https://huggingface.co/datasets/LeeAeron/F5TTSx/resolve/main/models/ESpeech-TTS-1_RL-V1/espeech_tts_rlv1.pt"
            )
            vocab = safe_path(
                "models/ESpeech/ESpeech-TTS-1_RL-V1/vocab.txt",
                "https://huggingface.co/datasets/LeeAeron/F5TTSx/resolve/main/models/ESpeech-TTS-1_RL-V1/vocab.txt"
            )
            tts_model_choice = ("ESpeech-TTS-1_RL-V1", ckpt, vocab,
                                json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2,
                                                text_dim=512, conv_layers=4)))
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=True)

        elif new_choice == "ESpeech-TTS-1_RL-V2":
            ckpt = safe_path(
                "models/ESpeech/ESpeech-TTS-1_RL-V2/espeech_tts_rlv2.pt",
                "https://huggingface.co/datasets/LeeAeron/F5TTSx/resolve/main/models/ESpeech-TTS-1_RL-V2/espeech_tts_rlv2.pt"
            )
            vocab = safe_path(
                "models/ESpeech/ESpeech-TTS-1_RL-V2/vocab.txt",
                "https://huggingface.co/datasets/LeeAeron/F5TTSx/resolve/main/models/ESpeech-TTS-1_RL-V2/vocab.txt"
            )
            tts_model_choice = ("ESpeech-TTS-1_RL-V2", ckpt, vocab,
                                json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2,
                                                text_dim=512, conv_layers=4)))
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=True)

        elif new_choice == "ESpeech-TTS-1_SFT-95K":
            ckpt = safe_path(
                "models/ESpeech/ESpeech-TTS-1_SFT-95K/espeech_tts_95k.pt",
                "https://huggingface.co/datasets/LeeAeron/F5TTSx/resolve/main/models/ESpeech-TTS-1_SFT-95K/espeech_tts_95k.pt"
            )
            vocab = safe_path(
                "models/ESpeech/ESpeech-TTS-1_SFT-95K/vocab.txt",
                "https://huggingface.co/datasets/LeeAeron/F5TTSx/resolve/main/models/ESpeech-TTS-1_SFT-95K/vocab.txt"
            )
            tts_model_choice = ("ESpeech-TTS-1_SFT-95K", ckpt, vocab,
                                json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2,
                                                text_dim=512, conv_layers=4)))
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=True)

        elif new_choice == "ESpeech-TTS-1_SFT-256K":
            ckpt = safe_path(
                "models/ESpeech/ESpeech-TTS-1_SFT-256K/espeech_tts_256k.pt",
                "https://huggingface.co/datasets/LeeAeron/F5TTSx/resolve/main/models/ESpeech-TTS-1_SFT-256K/espeech_tts_256k.pt"
            )
            vocab = safe_path(
                "models/ESpeech/ESpeech-TTS-1_SFT-256K/vocab.txt",
                "https://huggingface.co/datasets/LeeAeron/F5TTSx/resolve/main/models/ESpeech-TTS-1_SFT-256K/vocab.txt"
            )
            tts_model_choice = ("ESpeech-TTS-1_SFT-256K", ckpt, vocab,
                                json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2,
                                                text_dim=512, conv_layers=4)))
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=True)

        else:  # F5-TTS_v1
            ckpt = safe_path(
                "models/F5TTS_v1_Base/F5TTS_v1_Base_1250000.safetensors",
                "https://huggingface.co/datasets/LeeAeron/F5TTSx/resolve/main/models/F5TTS_v1_Base/F5TTS_v1_Base_1250000.safetensors"
            )
            vocab = safe_path(
                "models/F5TTS_v1_Base/vocab.txt",
                "https://huggingface.co/datasets/LeeAeron/F5TTSx/resolve/main/models/F5TTS_v1_Base/vocab.txt"
            )
            tts_model_choice = ("F5-TTS_v1", ckpt, vocab, DEFAULT_TTS_MODEL_CFG[2])
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=False)



    def set_custom_model(custom_ckpt_path, custom_vocab_path, custom_model_cfg):
        global tts_model_choice
        tts_model_choice = ("Custom", custom_ckpt_path, custom_vocab_path, custom_model_cfg)
        with open(last_used_custom, "w", encoding="utf-8") as f:
            f.write(custom_ckpt_path + "\n" + custom_vocab_path + "\n" + custom_model_cfg + "\n")

    from f5_tts.infer.utils_infer import (
        load_last_used_asr,
        set_asr_model,
        initialize_asr_pipeline,
    )


    with gr.Row():
        choose_tts_model = gr.Radio(
            choices=[
                "F5-TTS_v1",
                "Misha24-10_v2",
                "Misha24-10_v4",
                "ESpeech-TTS-1_podcaster",
                "ESpeech-TTS-1_RL-V1",
                "ESpeech-TTS-1_RL-V2",
                "ESpeech-TTS-1_SFT-95K",
                "ESpeech-TTS-1_SFT-256K",
                "Custom"
            ],
            label="Choose TTS Model",
            value="F5-TTS_v1",  # default native
        )

        custom_ckpt_path = gr.Dropdown(
            choices=[DEFAULT_TTS_MODEL_CFG[0]],
            value=DEFAULT_TTS_MODEL_CFG[0],
            allow_custom_value=True,
            label="Model: local_path | hf://user_id/repo_id/model_ckpt",
            visible=False,
        )
        custom_vocab_path = gr.Dropdown(
            choices=[DEFAULT_TTS_MODEL_CFG[1]],
            value=DEFAULT_TTS_MODEL_CFG[1],
            allow_custom_value=True,
            label="Vocab: local_path | hf://user_id/repo_id/vocab_file",
            visible=False,
        )
        custom_model_cfg = gr.Dropdown(
            choices=[
                DEFAULT_TTS_MODEL_CFG[2],
                json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2,
                                text_dim=512, text_mask_padding=False,
                                conv_layers=4, pe_attn_head=1)),
                json.dumps(dict(dim=768, depth=18, heads=12, ff_mult=2,
                                text_dim=512, text_mask_padding=False,
                                conv_layers=4, pe_attn_head=1)),
            ],
            value=DEFAULT_TTS_MODEL_CFG[2],
            allow_custom_value=True,
            label="Config: in a dictionary form",
            visible=False,
        )

        # --- ASR ---
        with gr.Column(scale=1):
            asr_status = gr.Textbox(
                label="ASR Status: Tiny < Base < Small < Medium < Large (Gb)",
                interactive=False,
                visible=False,
            )
            choose_asr_model = gr.Dropdown(
                choices=[
                    "openai/whisper-tiny",
                    "openai/whisper-base",
                    "openai/whisper-small",
                    "openai/whisper-medium",
                    "openai/whisper-large-v3-turbo"
                ],
                label="Whisper ASR Model",
                value=load_last_used_asr(),
            )

    choose_tts_model.change(
        switch_tts_model,
        inputs=[choose_tts_model],
        outputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg, use_accent_checkbox],
        show_progress="hidden",
    )

    custom_ckpt_path.change(
        set_custom_model,
        inputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )
    custom_vocab_path.change(
        set_custom_model,
        inputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )
    custom_model_cfg.change(
        set_custom_model,
        inputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )

    choose_asr_model.change(
        set_asr_model,
        inputs=[choose_asr_model],
        outputs=[asr_status],
        show_progress="hidden",
    )

    # --- tabs ---
    gr.TabbedInterface(
        [app_tts, app_multistyle, app_chat, app_additional],
        ["Basic-TTS", "Multi-Speech", "Voice-Chat", "Additional"],
    )
    
    gr.Markdown(
        f"""
If you're having issues, try converting your reference audio to WAV or MP3, clipping it to 12s with  ✂  in the bottom right corner (otherwise might have non-optimal auto-trimmed result).

**NOTE: Reference text will be automatically transcribed with Whisper if not provided. For best results, keep your reference clips short (<12s). Ensure the audio is fully uploaded before generating.**
"""
    )

@click.command()
@click.option("--port", "-p", default=None, type=int, help="Port to run the app on")
@click.option("--host", "-H", default=None, help="Host to run the app on")
@click.option(
    "--share",
    "-s",
    default=False,
    is_flag=True,
    help="Share the app via Gradio share link",
)
@click.option("--api", "-a", default=True, is_flag=True, help="Allow API access")
@click.option(
    "--root_path",
    "-r",
    default=None,
    type=str,
    help='The root path (or "mount point") of the application, if it\'s not served from the root ("/") of the domain. Often used when the application is behind a reverse proxy that forwards requests to the application, e.g. set "/myapp" or full URL for application served at "https://example.com/myapp".',
)
@click.option(
    "--inbrowser",
    "-i",
    is_flag=True,
    default=True,
    help="Automatically launch the interface in the default web browser",
)
def main(port, host, share, api, root_path, inbrowser):
    global app
    print("Starting app...")
    app.queue(api_open=api).launch(
        server_name=host,
        server_port=port,
        share=share,
        root_path=root_path,
        inbrowser=inbrowser,
    )


if __name__ == "__main__":
    if not USING_SPACES:
        main()
    else:
        app.queue().launch()
