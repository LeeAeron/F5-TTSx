# ruff: noqa: E402
# Above allows ruff to ignore E402: module level import not at top of file
import asyncio
import sys
import logging
import threading
import time
import webbrowser
import uvicorn
from PIL import Image

# Spectrogram imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
import librosa.display


if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

import gc
import json
import os
import re
import tempfile
import requests
from collections import OrderedDict
import functools
from functools import lru_cache
from importlib.resources import files
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Iterator, Optional

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

# Smart chunking imports
import random
import hashlib
import pickle

# Chunking modes
CHUNKING_MODES = ["lines", "sentences", "characters"]

# Crossfade range (ms)
CROSSFADE_MS_RANGE = [0, 10, 20, 30, 40, 50, 60, 80, 100, 150, 200]

# Max lines per chunk options
MAX_LINES_OPTIONS = list(range(1, 21))

# Max sentences per chunk options
MAX_SENTENCES_OPTIONS = list(range(1, 21))

# Max chars per chunk options
MAX_CHARS_OPTIONS = list(range(100, 3001, 50))

# Audio processing constants
AUDIO_FORMATS = ["wav", "mp3", "flac", "ogg", "opus", "m4a", "aac"]
OUTPUT_SAMPLE_RATES = [24000, 32000, 44100, 48000]
BITRATE_OPTIONS = [64, 96, 128, 160, 192, 224, 256, 320]
OGG_QUALITY_OPTIONS = list(range(-1, 11))
NORMALIZATION_LEVELS = [-20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]



# Sentence splitting regex - matches sentence endings: . ! ? ... (with optional closing quotes/brackets)
SENTENCE_END_PATTERN = re.compile(
    r'[.!?]+[\'"\")\]]*\s+|[.!?]+[\'"\")\]]*$',
    re.MULTILINE
)


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences, preserving sentence delimiters."""
    if not text.strip():
        return []
    # Split by sentence endings but keep the delimiters
    parts = SENTENCE_END_PATTERN.split(text)
    # Filter empty strings
    sentences = [s.strip() for s in parts if s.strip()]
    return sentences



@dataclass
class ScriptChunk:
    """Represents a single chunk of text for generation."""
    text: str
    start_idx: int
    end_idx: int
    is_first: bool
    is_last: bool
    estimated_duration_sec: float = 0.0


class ScriptChunker:
    """Splits long text into adaptively sized chunks.

    Supports three chunking modes:
    - "lines": chunk by number of lines
    - "sentences": chunk by number of sentences (detects . ! ? ... endings)
    - "characters": chunk by character count
    """

    def __init__(
        self,
        chunking_mode: str = "lines",
        max_lines_per_chunk: int = 8,
        max_sentences_per_chunk: int = 5,
        max_chars_per_chunk: int = 800,
        max_tokens_estimate: int = 1500,
        chars_per_token: float = 3.5,
        avg_words_per_minute: float = 150.0,
    ):
        self.chunking_mode = chunking_mode
        self.max_lines = max_lines_per_chunk
        self.max_sentences = max_sentences_per_chunk
        self.max_chars = max_chars_per_chunk
        self.max_tokens = max_tokens_estimate
        self.chars_per_token = chars_per_token
        self.wpm = avg_words_per_minute


    def _estimate_tokens(self, text: str) -> int:
        return int(len(text) / self.chars_per_token)

    def _estimate_duration(self, text: str) -> float:
        words = len(text.split())
        return (words / self.wpm) * 60.0



    def _chunk_by_lines(self, lines: List[str]) -> List[ScriptChunk]:
        """Original line-based chunking."""
        if not lines:
            return []

        chunks = []
        start_idx = 0

        while start_idx < len(lines):
            end_idx = start_idx + 1
            current_chars = len(lines[start_idx])
            current_tokens = self._estimate_tokens(lines[start_idx])

            while (end_idx < len(lines) and 
                   end_idx - start_idx < self.max_lines and
                   current_tokens < self.max_tokens and
                   current_chars < self.max_chars):

                next_line = lines[end_idx]
                next_tokens = self._estimate_tokens(next_line)

                if current_tokens + next_tokens > self.max_tokens * 1.1:
                    break

                current_chars += len(next_line)
                current_tokens += next_tokens
                end_idx += 1

            chunk_text = "\n".join(lines[start_idx:end_idx])
            est_duration = self._estimate_duration(chunk_text)

            chunks.append(ScriptChunk(
                text=chunk_text,
                start_idx=start_idx,
                end_idx=end_idx - 1,
                is_first=(start_idx == 0),
                is_last=(end_idx >= len(lines)),
                estimated_duration_sec=est_duration
            ))

            start_idx = end_idx

        return chunks

    def _chunk_by_sentences(self, text: str) -> List[ScriptChunk]:
        """Chunk by number of sentences."""
        sentences = split_into_sentences(text)
        if not sentences:
            return []

        chunks = []
        current_sentences = []
        current_chars = 0
        current_tokens = 0
        start_idx = 0
        sent_idx = 0

        for sentence in sentences:
            sent_chars = len(sentence)
            sent_tokens = self._estimate_tokens(sentence)

            would_exceed = (
                len(current_sentences) >= self.max_sentences or
                current_chars + sent_chars > self.max_chars or
                current_tokens + sent_tokens > self.max_tokens
            )

            if would_exceed and current_sentences:
                chunk_text = " ".join(current_sentences)
                chunks.append(ScriptChunk(
                    text=chunk_text,
                    start_idx=start_idx,
                    end_idx=sent_idx - 1,
                    is_first=(start_idx == 0),
                    is_last=False,
                    estimated_duration_sec=self._estimate_duration(chunk_text)
                ))
                start_idx = sent_idx
                current_sentences = [sentence]
                current_chars = sent_chars
                current_tokens = sent_tokens
            else:
                current_sentences.append(sentence)
                current_chars += sent_chars
                current_tokens += sent_tokens

            sent_idx += 1

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(ScriptChunk(
                text=chunk_text,
                start_idx=start_idx,
                end_idx=sent_idx - 1,
                is_first=(start_idx == 0),
                is_last=True,
                estimated_duration_sec=self._estimate_duration(chunk_text)
            ))

        return chunks

    def _chunk_by_characters(self, text: str) -> List[ScriptChunk]:
        """Chunk by total character count, never cutting inside a word."""
        if not text:
            return []

        chunks = []
        text_len = len(text)
        start = 0
        char_idx = 0
        
        # Minimum tail length at which don't chop the word
        MIN_TAIL_LENGTH = 10  # If there are < 10 characters left to the end, don't look for a space.
        
        while start < text_len:
            end = min(start + self.max_chars, text_len)
            
            # If it's not the end of the text and you're in the middle of a word, roll back to the word boundary.
            if end < text_len:
                # check if it is in the middle of a word
                if text[end] not in ' \t\n\r':
                    # search for the nearest space back (at least 20% of max_chars)
                    min_backtrack = max(start + int(self.max_chars * 0.2), start + 1)
                    space_pos = text.rfind(' ', min_backtrack, end + 1)
                    
                    if space_pos > start:
                        end = space_pos
                    else:
                        # No space - look for a line break
                        newline_pos = text.rfind('\n', start, end + 1)
                        if newline_pos > start:
                            end = newline_pos
                        else:
                            # If there is no transfer, chop it, but take at least 20% max_chars
                            if end - start > self.max_chars * 0.8:
                                # look for punctuation as an emergency breaking point
                                punct_pos = -1
                                for punct in ',;:-':
                                    pos = text.rfind(punct, min_backtrack, end + 1)
                                    if pos > punct_pos:
                                        punct_pos = pos
                                if punct_pos > start:
                                    end = punct_pos + 1  # include punctuation marks
            
            chunk_text = text[start:end].strip()
            
            # If after trimming by space the result is empty, take it forcibly
            if not chunk_text and start < text_len:
                forced_end = min(start + self.max_chars, text_len)
                chunk_text = text[start:forced_end].strip()
                end = forced_end
            
            if chunk_text:
                chunks.append(ScriptChunk(
                    text=chunk_text,
                    start_idx=start,
                    end_idx=end - 1,
                    is_first=(start == 0),
                    is_last=(end >= text_len),
                    estimated_duration_sec=self._estimate_duration(chunk_text)
                ))
            
            # Move start beyond the end of the current chunk (skip the space)
            start = end
            # Skipping leading spaces/newlines for the next chunk
            while start < text_len and text[start] in ' \t\n\r':
                start += 1

        return chunks

    def parse_text(self, text: str) -> List[ScriptChunk]:
        """Parse text into chunks using the configured chunking mode."""
        if not text or not text.strip():
            return []

        if self.chunking_mode == "sentences":
            return self._chunk_by_sentences(text)
        elif self.chunking_mode == "characters":
            return self._chunk_by_characters(text)
        else:  # "lines" (default)
            lines = [l for l in text.strip().split("\n") if l.strip()]
            return self._chunk_by_lines(lines)


class AudioCrossfader:
    """Smooth gluing of audio chunks with cosine crossfade."""

    def __init__(self, fade_duration_ms: float = 50.0, sample_rate: int = 24000):
        self.fade_samples = max(1, int(sample_rate * (fade_duration_ms / 1000.0)))
        self.sample_rate = sample_rate

    def apply_crossfade(self, chunk1: np.ndarray, chunk2: np.ndarray) -> np.ndarray:
        if len(chunk1) < self.fade_samples * 2 or len(chunk2) < self.fade_samples * 2:
            return np.concatenate([chunk1, chunk2])

        fade_out = chunk1[-self.fade_samples:]
        fade_in = chunk2[:self.fade_samples]

        t = np.linspace(0, 1, self.fade_samples)
        curve_out = np.cos(t * np.pi / 2)
        curve_in = np.sin(t * np.pi / 2)

        mixed = fade_out * curve_out + fade_in * curve_in

        result = np.concatenate([
            chunk1[:-self.fade_samples],
            mixed,
            chunk2[self.fade_samples:]
        ])

        return result

    def concatenate_chunks(self, chunks: List[np.ndarray]) -> np.ndarray:
        if not chunks:
            return np.array([])
        if len(chunks) == 1:
            return chunks[0]

        result = chunks[0]
        for next_chunk in chunks[1:]:
            result = self.apply_crossfade(result, next_chunk)

        return result


@dataclass
class GenerationCheckpoint:
    """Save point to resume generation."""
    session_id: str
    text_hash: str
    num_step: int
    guidance_scale: float
    speed: float
    seed: int
    # state
    total_chunks: int
    completed_chunks: int
    chunk_audio_files: List[str]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "text_hash": self.text_hash,
            "num_step": self.num_step,
            "guidance_scale": self.guidance_scale,
            "speed": self.speed,
            "seed": self.seed,
            "total_chunks": self.total_chunks,
            "completed_chunks": self.completed_chunks,
            "chunk_audio_files": self.chunk_audio_files,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GenerationCheckpoint":
        return cls(**data)


class CheckpointManager:
    """Controls saving and loading checkpoints."""

    def __init__(self, checkpoint_dir: Path = None):
        if checkpoint_dir is None:
            checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _get_checkpoint_path(self, session_id: str) -> Path:
        return self.checkpoint_dir / f"checkpoint_{session_id}.pkl"

    def save(self, checkpoint: GenerationCheckpoint):
        path = self._get_checkpoint_path(checkpoint.session_id)
        with open(path, 'wb') as f:
            pickle.dump(checkpoint.to_dict(), f)
        logging.info(f"[Checkpoint] Saved: {path} ({checkpoint.completed_chunks}/{checkpoint.total_chunks} chunks)")

    def load(self, session_id: str) -> Optional[GenerationCheckpoint]:
        path = self._get_checkpoint_path(session_id)
        if not path.exists():
            return None
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            ckpt = GenerationCheckpoint.from_dict(data)
            logging.info(f"[Checkpoint] Loaded: {session_id} ({ckpt.completed_chunks}/{ckpt.total_chunks} chunks)")
            return ckpt
        except Exception as e:
            logging.error(f"[Checkpoint] Failed to load: {e}")
            return None

    def delete(self, session_id: str):
        path = self._get_checkpoint_path(session_id)
        if path.exists():
            os.remove(path)
            logging.info(f"[Checkpoint] Deleted: {session_id}")

    def list_available(self) -> List[Tuple[str, int, int, float]]:
        """Returns a list (session_id, completed, total, timestamp)."""
        available = []
        for fname in os.listdir(self.checkpoint_dir):
            if fname.startswith("checkpoint_") and fname.endswith(".pkl"):
                try:
                    with open(self.checkpoint_dir / fname, 'rb') as f:
                        data = pickle.load(f)
                    sid = data["session_id"]
                    available.append((sid, data["completed_chunks"], data["total_chunks"], data["timestamp"]))
                except:
                    continue
        return sorted(available, key=lambda x: x[3], reverse=True)

    def generate_session_id(self, text: str, seed: int) -> str:
        """Generates a unique session ID based on parameters."""
        effective_seed = seed if seed >= 0 else 0
        content = f"{text}:{effective_seed}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class ProgressTracker:
    """Tracks progress and calculates ETA."""

    def __init__(self, total_chunks: int):
        self.total = total_chunks
        self.completed = 0
        self.chunk_times: List[float] = []
        self.start_time = time.time()
        self.current_chunk_start = 0.0

    def start_chunk(self):
        self.current_chunk_start = time.time()

    def finish_chunk(self):
        elapsed = time.time() - self.current_chunk_start
        self.chunk_times.append(elapsed)
        self.completed += 1

    def get_eta_seconds(self) -> float:
        if not self.chunk_times:
            return 0.0
        avg_time = sum(self.chunk_times) / len(self.chunk_times)
        remaining = self.total - self.completed
        return avg_time * remaining

    def get_progress_percent(self) -> float:
        if self.total == 0:
            return 100.0
        return (self.completed / self.total) * 100.0

    def get_stats(self) -> str:
        elapsed = time.time() - self.start_time
        eta = self.get_eta_seconds()
        pct = self.get_progress_percent()

        elapsed_str = self._format_time(elapsed)
        eta_str = self._format_time(eta)

        if self.chunk_times:
            avg_chunk = sum(self.chunk_times) / len(self.chunk_times)
            avg_str = f"{avg_chunk:.1f}s/chunk"
        else:
            avg_str = "calculating..."

        bar_len = 20
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)

        return f"[{bar}] {pct:.0f}% | {self.completed}/{self.total} chunks | Elapsed: {elapsed_str} | ETA: {eta_str} | {avg_str}"

    def _format_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"


def generate_progress_html(progress_tracker: ProgressTracker, current_chunk: int = None) -> str:
    """Generates HTML progress bar in unified dark-purple theme."""
    pct = progress_tracker.get_progress_percent()
    stats = progress_tracker.get_stats()

    gradient = "linear-gradient(90deg, #667eea 0%, #764ba2 100%)"
    bg_dark = "#0f172a"
    bg_card = "#1e293b"
    border_color = "#334155"
    text_primary = "#e2e8f0"
    text_secondary = "#94a3b8"
    accent = "#667eea"

    if pct >= 100:
        status_color = "#10b981"
        status_icon = "✅"
    elif pct > 0:
        status_color = accent
        status_icon = "🔄"
    else:
        status_color = "#64748b"
        status_icon = "⏳"

    chunk_indicator = ""
    if current_chunk is not None and progress_tracker.total > 1:
        chunk_indicator = f'<span style="color:{accent}; font-size:13px; font-weight:600;">{status_icon} Chunk {current_chunk}/{progress_tracker.total}</span>'

    html = f"""
    <div style="width:100%; background:{bg_dark}; border-radius:12px; padding:16px; margin:8px 0; 
                border:1px solid {border_color}; font-family:'Segoe UI',system-ui,sans-serif;
                box-shadow:0 4px 6px rgba(0,0,0,0.3);">
        <div style="display:flex; justify-content:space-between; margin-bottom:10px; align-items:center;">
            <span style="color:{text_primary}; font-size:14px; font-weight:600;">🎙️ Generation Progress</span>
            <div style="display:flex; gap:12px; align-items:center;">
                {chunk_indicator}
                <span style="color:{text_secondary}; font-size:13px;">{pct:.0f}% complete</span>
            </div>
        </div>
        <div style="width:100%; height:24px; background:{bg_card}; border-radius:12px; overflow:hidden;
                    box-shadow:inset 0 2px 4px rgba(0,0,0,0.3);">
            <div style="width:{pct}%; height:100%; background:{gradient}; 
                        transition:width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
                        border-radius:12px; position:relative;">
                <div style="position:absolute; right:0; top:0; bottom:0; width:30px; 
                            background:linear-gradient(90deg, transparent, rgba(255,255,255,0.3));"></div>
            </div>
        </div>
        <div style="display:flex; justify-content:space-between; margin-top:8px; flex-wrap:wrap; gap:8px;">
            <span style="color:{text_secondary}; font-size:12px; font-family:monospace;">{stats}</span>
            <span style="color:{status_color}; font-size:12px; font-weight:600;">
                {status_icon} {"Complete" if pct >= 100 else "In Progress" if pct > 0 else "Waiting"}
            </span>
        </div>
    </div>
    """
    return html



# Audio normalization

def normalize_audio(waveform: np.ndarray, target_level_db: float) -> np.ndarray:
    """
    Normalize audio to target RMS level in dB.
    target_level_db: -20 to +10 dB relative to full scale.
    """
    if waveform.size == 0:
        return waveform

    # Convert to float
    if waveform.dtype == np.int16:
        waveform_float = waveform.astype(np.float32) / 32768.0
    else:
        waveform_float = waveform.astype(np.float32)

    # Calculate current RMS
    rms = np.sqrt(np.mean(waveform_float ** 2))
    if rms < 1e-10:
        return waveform

    # Target RMS from dB
    target_rms = 10 ** (target_level_db / 20.0)

    # Gain factor
    gain = target_rms / rms

    # Apply gain
    normalized = waveform_float * gain

    # Clip to prevent overflow
    normalized = np.clip(normalized, -1.0, 1.0)

    # Convert back to int16
    if waveform.dtype == np.int16:
        return (normalized * 32767).astype(np.int16)

    return normalized

# Settings management
SETTINGS_FILE = Path("f5ttsx_settings.json")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)


def load_settings() -> Dict[str, Any]:
    """Load settings from f5ttsx_settings.json."""
    defaults = {
        # Basic TTS settings
        "speed": 1.0,
        "nfe_step": 16,
        "cross_fade_duration": 0.0,
        "target_rms": 0.1,
        "cfg_strength": 1.0,
        "use_accent": True,
        "remove_silence": False,
        "seed": -1,
        "randomize_seed": True,
        "output_format": "mp3",
        "output_sample_rate": 48000,
        "bitrate": 320,
        "ogg_quality": 5,
        # Audio processing
        "normalize": True,
        "normalize_level": -15,
        # Chunking settings
        "enable_chunking": True,
        "chunking_mode": "lines",
        "max_lines_per_chunk": 8,
        "max_sentences_per_chunk": 5,
        "max_chars_per_chunk": 800,
        "crossfade_ms": 200,
        "clear_checkpoints_on_startup": True,
    }

    if not SETTINGS_FILE.exists():
        return defaults

    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
        # Merge with defaults
        for key, val in defaults.items():
            if key not in saved:
                saved[key] = val
        return saved
    except Exception as e:
        logging.warning(f"Failed to load settings: {e}")
        return defaults


def save_settings(settings: Dict[str, Any]):
    """Save settings to f5ttsx_settings.json."""
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        logging.info(f"Settings saved to {SETTINGS_FILE}")
    except Exception as e:
        logging.error(f"Failed to save settings: {e}")


def cleanup_checkpoints_folder():
    """Remove all files from checkpoints folder."""
    try:
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            return 0
        removed = 0
        for item in os.listdir(CHECKPOINT_DIR):
            item_path = os.path.join(CHECKPOINT_DIR, item)
            try:
                if os.path.isfile(item_path):
                    os.remove(item_path)
                    removed += 1
                elif os.path.isdir(item_path):
                    import shutil
                    shutil.rmtree(item_path)
                    removed += 1
            except Exception as e:
                logging.warning(f"[CheckpointCleanup] Failed to delete {item}: {e}")
        logging.info(f"[CheckpointCleanup] Cleared {removed} items from {CHECKPOINT_DIR}")
        return removed
    except Exception as e:
        logging.error(f"[CheckpointCleanup] Error: {e}")
        return 0


# F5-TTSx 

def save_audio(final_wave, final_sample_rate, fmt="wav", used_seed=None, target_sr=48000, bitrate=320, ogg_quality=5):
    """Save audio via ffmpeg with configurable sample rate, bitrate, and OGG quality."""
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav_path = tmp_wav.name
    tmp_wav.close()  # Close the file handle so ffmpeg can access it on Windows

    sf.write(tmp_wav_path, final_wave, final_sample_rate)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    seed_str = str(used_seed) if used_seed is not None else "noseed"
    out_filename = f"tts_{timestamp}_{seed_str}.{fmt}"
    out_path = os.path.join("outputs", out_filename)

    # Build ffmpeg command with resampling and format-specific encoding
    cmd = ["ffmpeg", "-y", "-i", tmp_wav_path, "-ar", str(target_sr)]

    fmt = fmt.lower().replace(".", "")
    if fmt == "mp3":
        cmd.extend(["-c:a", "libmp3lame", "-b:a", f"{bitrate}k"])
    elif fmt == "ogg":
        # OGG Vorbis uses quality-based encoding (-q:a) rather than fixed bitrate
        cmd.extend(["-c:a", "libvorbis", "-q:a", str(ogg_quality)])
    elif fmt == "opus":
        # OPUS uses libopus with VBR bitrate
        cmd.extend(["-c:a", "libopus", "-b:a", f"{bitrate}k", "-vbr", "on"])
    elif fmt in ("m4a", "aac"):
        cmd.extend(["-c:a", "aac", "-b:a", f"{bitrate}k"])
    elif fmt == "flac":
        cmd.extend(["-c:a", "flac"])
    else:
        cmd.extend(["-c:a", "pcm_s16le"])

    cmd.append(out_path)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logging.warning(f"FFmpeg error: {result.stderr}. Falling back to wav.")
        out_path = out_path.replace(f".{fmt}", ".wav")
        sf.write(out_path, final_wave, final_sample_rate)

    try:
        os.unlink(tmp_wav_path)
    except PermissionError:
        pass  # Ignore if Windows still holds the file
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
    tempfile_kwargs,
)
from f5_tts.infer.utils_infer import (
    load_last_used_asr,
    set_asr_model,
    initialize_asr_pipeline,
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
                    speed = downloaded / elapsed if elapsed > 0 else 0
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


# basic settings
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


# Accentuation Support
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

# External dictionaries
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

# RUAccent + convert_accent_to_plus
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



# Spectrogram function
def save_spectrogram(waveform, sample_rate, output_path=None):
    """Save spectrogram image using librosa + matplotlib (VibeVoice style)."""
    if output_path is None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            output_path = tmp.name
    fig, ax = plt.subplots(figsize=(12, 4))
    if hasattr(waveform, 'dtype') and waveform.dtype == np.int16:
        waveform_float = waveform.astype(np.float32) / 32768.0
    elif hasattr(waveform, 'cpu'):
        waveform_float = waveform.cpu().numpy().astype(np.float32)
        if waveform_float.max() > 1.0 or waveform_float.min() < -1.0:
            waveform_float = waveform_float / max(abs(waveform_float.max()), abs(waveform_float.min()))
    else:
        waveform_float = np.array(waveform).astype(np.float32)
        if waveform_float.max() > 1.0 or waveform_float.min() < -1.0:
            waveform_float = waveform_float / max(abs(waveform_float.max()), abs(waveform_float.min()))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(waveform_float)), ref=np.max)
    img = librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='hz', ax=ax)
    ax.set_title('Spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    return output_path


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
    )

    # Remove silence
    if remove_silence:
        with tempfile.NamedTemporaryFile(suffix=".wav", **tempfile_kwargs) as f:
            temp_path = f.name
        try:
            sf.write(temp_path, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = load_audio(f.name, target_sample_rate=final_sample_rate)
            if not isinstance(final_wave, torch.Tensor):
                final_wave = torch.from_numpy(final_wave).float()
        finally:
            os.unlink(temp_path)
        final_wave = final_wave.squeeze().cpu().numpy()

    # Save the spectrogram (VibeVoice style)
    with tempfile.NamedTemporaryFile(suffix=".png", **tempfile_kwargs) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
    save_spectrogram(final_wave, final_sample_rate, spectrogram_path)

    final_wave = (final_wave * 32767).astype(np.int16)

    return (final_sample_rate, final_wave), spectrogram_path, ref_text, used_seed


# Chunked generation for Basic-TTS

def _set_seed(seed_value: int):
    """Set PyTorch random seed for reproducible generation."""
    if seed_value is not None and int(seed_value) >= 0:
        seed = int(seed_value)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        return seed
    else:
        seed = random.randint(0, 2**32 - 1)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        return seed


def _generate_single_chunk(
    ref_audio,
    ref_text,
    gen_text,
    model,
    remove_silence,  # ignored for chunks — applied to final result only
    seed,
    cross_fade_duration,
    nfe_step,
    speed,
    use_accent,
    target_rms,
    cfg_strength,
):
    """Generate audio for a single text chunk (wrapper around infer)."""
    audio_out, spectrogram_path, ref_text_out, used_seed = infer(
        ref_audio,
        ref_text,
        gen_text,
        model,
        remove_silence=False,  # NEVER remove silence for individual chunks
        seed=seed,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        speed=speed,
        use_accent=use_accent,
        target_rms=target_rms,
        cfg_strength=cfg_strength,
        show_info=print,
    )
    return audio_out, spectrogram_path, ref_text_out, used_seed



def _spectrogram_to_html(image_path: str) -> str:
    """Convert spectrogram image path to clickable HTML with fullscreen modal."""
    if not image_path or not os.path.exists(image_path):
        return ""

    import base64
    with open(image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode()

    # Determine MIME type
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/png" if ext == ".png" else "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"

    html = f"""
    <div style="width:100%; cursor:pointer;" onclick="document.getElementById('spec-modal').style.display='flex'">
        <img src="data:{mime};base64,{img_data}" 
             style="width:100%; max-height:300px; object-fit:contain; border-radius:8px;"
             title="Click to enlarge">
    </div>
    <div id="spec-modal" style="display:none; position:fixed; top:0; left:0; width:100vw; height:100vh; 
                                background:rgba(0,0,0,0.95); z-index:9999; align-items:center; justify-content:center;"
         onclick="this.style.display='none'">
        <img src="data:{mime};base64,{img_data}" 
             style="max-width:95vw; max-height:95vh; object-fit:contain; cursor:zoom-out;">
    </div>
    """
    return html


@gpu_decorator
def generate_chunked_tts(
    ref_audio_input,
    ref_text_input,
    gen_text_input,
    model,
    remove_silence,
    seed_input,
    cross_fade_duration_slider,
    nfe_slider,
    speed_slider,
    use_accent,
    target_rms_input,
    cfg_strength_input,
    output_format,
    # Audio processing
    normalize=True,
    normalize_level=-15,
    # Chunking parameters
    enable_chunking=True,
    chunking_mode="lines",
    max_lines_per_chunk=8,
    max_sentences_per_chunk=5,
    max_chars_per_chunk=800,
    crossfade_ms=200,
    target_sample_rate=48000,
    bitrate=320,
    ogg_quality=5,
):
    """Chunked generation with progress tracking for Basic-TTS."""

    if not gen_text_input or not gen_text_input.strip():
        yield None, "Enter text to be synthesized", None, None, ""
        return

    if not ref_audio_input:
        yield None, "Please provide reference audio.", None, None, ""
        return

    # Set seed
    actual_seed = _set_seed(seed_input)

    # Check if chunking should be used
    should_chunk = False
    if enable_chunking:
        if chunking_mode == "lines":
            total_lines = len([l for l in gen_text_input.strip().split("\n") if l.strip()])
            should_chunk = total_lines > max_lines_per_chunk
        elif chunking_mode == "sentences":
            total_sentences = len(split_into_sentences(gen_text_input))
            should_chunk = total_sentences > max_sentences_per_chunk
        elif chunking_mode == "characters":
            total_chars = len(gen_text_input.strip())
            should_chunk = total_chars > max_chars_per_chunk

    # Single-pass generation (no chunking needed)
    if not should_chunk:
        try:
            audio_out, spectrogram_path, ref_text_out, used_seed = _generate_single_chunk(
                ref_audio_input,
                ref_text_input,
                gen_text_input,
                model,
                remove_silence,
                actual_seed,
                cross_fade_duration_slider,
                nfe_slider,
                speed_slider,
                use_accent,
                target_rms_input,
                cfg_strength_input,
            )

            sr, audio_data = audio_out

            # Apply normalization if enabled
            if normalize:
                audio_data = normalize_audio(audio_data, float(normalize_level))

            out_path = save_audio(audio_data, sr, output_format, used_seed, target_sr=target_sample_rate, bitrate=bitrate, ogg_quality=ogg_quality)

            status_msg = f"Done! Saved in: {Path(out_path).name} | Seed: {used_seed}"
            # Convert spectrogram to HTML with click-to-enlarge
            spectrogram_html = _spectrogram_to_html(spectrogram_path) if spectrogram_path else ""
            yield audio_out, status_msg, out_path, spectrogram_html, ""

        except Exception as e:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            yield None, f"Error: {type(e).__name__}: {str(e)[:100]}", None, None, ""
        return

    # Chunked generation
    chunker = ScriptChunker(
        chunking_mode=chunking_mode,
        max_lines_per_chunk=max_lines_per_chunk,
        max_sentences_per_chunk=max_sentences_per_chunk,
        max_chars_per_chunk=max_chars_per_chunk,
    )
    # Use model's actual sample rate (typically 24000 for F5-TTS)
    crossfader = AudioCrossfader(fade_duration_ms=crossfade_ms, sample_rate=24000)

    chunks = chunker.parse_text(gen_text_input)
    total_chunks = len(chunks)

    if total_chunks == 0:
        yield None, "Error: Could not parse text into chunks", None, None, ""
        return

    # Initialize checkpoint manager
    checkpoint_mgr = CheckpointManager()
    session_id = checkpoint_mgr.generate_session_id(gen_text_input, actual_seed)
    ckpt = GenerationCheckpoint(
        session_id=session_id,
        text_hash=hashlib.md5(gen_text_input.encode()).hexdigest(),
        num_step=nfe_slider,
        guidance_scale=cfg_strength_input,
        speed=speed_slider,
        seed=actual_seed,
        total_chunks=total_chunks,
        completed_chunks=0,
        chunk_audio_files=[],
    )

    all_chunk_audios = []
    accumulated_audio = None
    sr = 24000  # F5-TTS default sample rate
    
    last_spectrogram_image = None

    # Progress tracker
    progress = ProgressTracker(total_chunks)

    status_msg = f"📦 Chunked generation: {total_chunks} chunks"
    progress_html = generate_progress_html(progress)
    yield None, status_msg, None, None, progress_html

    # Generate all chunks
    for i, chunk in enumerate(chunks):
        chunk_idx = i
        progress.start_chunk()

        # Show "processing" status with progress BEFORE blocking call
        chunk_status = f"🔄 Processing chunk {chunk_idx + 1}/{total_chunks}..."
        progress_html = generate_progress_html(progress, chunk_idx + 1)
        yield None, chunk_status, None, None, progress_html
        # Give Gradio time to render - CRITICAL for progress bar visibility
        time.sleep(0.15)

        try:
            audio_out, spectrogram_path, ref_text_out, used_seed = _generate_single_chunk(
                ref_audio_input,
                ref_text_input,
                chunk.text,
                model,
                remove_silence,
                actual_seed,
                cross_fade_duration_slider,
                nfe_slider,
                speed_slider,
                use_accent,
                target_rms_input,
                cfg_strength_input,
            )
            sr, chunk_waveform = audio_out
            
            try:
                last_spectrogram_image = Image.open(spectrogram_path)
            except Exception:
                last_spectrogram_image = None
        except Exception as e:
            logging.error(f"[Chunk {chunk_idx + 1}] Error: {e}")
            progress.finish_chunk()
            chunk_status = f"❌ Chunk {chunk_idx + 1} failed: {e}"
            progress_html = generate_progress_html(progress, chunk_idx + 1)
            yield None, chunk_status, None, None, progress_html
            continue

        if chunk_waveform is None or len(chunk_waveform) == 0:
            progress.finish_chunk()
            chunk_status = f"⚠️ Chunk {chunk_idx + 1} produced no audio"
            progress_html = generate_progress_html(progress, chunk_idx + 1)
            yield None, chunk_status, None, None, progress_html
            continue

        # Convert int16 from infer() to float32 [-1.0, 1.0] for processing
        if chunk_waveform.dtype == np.int16:
            chunk_float = chunk_waveform.astype(np.float32) / 32768.0
        else:
            chunk_float = chunk_waveform.astype(np.float32)

        # Save chunk for resume (save original int16)
        chunk_filename = f"chunk_{session_id}_{chunk_idx:04d}.wav"
        chunk_path = CHECKPOINT_DIR / chunk_filename
        sf.write(chunk_path, chunk_waveform, sr, subtype='PCM_16')

        all_chunk_audios.append(chunk_float)
        ckpt.chunk_audio_files.append(str(chunk_path))
        ckpt.completed_chunks = len(all_chunk_audios)
        checkpoint_mgr.save(ckpt)

        progress.finish_chunk()

        # Incremental accumulation with crossfade for ALL modes
        if accumulated_audio is None:
            accumulated_audio = chunk_float.copy()
        else:
            accumulated_audio = crossfader.apply_crossfade(accumulated_audio, chunk_float)

        progress_html = generate_progress_html(progress, chunk_idx + 1)
        chunk_status = f"✅ Chunk {chunk_idx + 1}/{total_chunks}: {len(chunk_waveform)/sr:.1f}s | Total: {len(accumulated_audio)/sr:.1f}s"

        # Stream intermediate result - convert float32 back to int16 for audio player
        audio_int16 = (np.clip(accumulated_audio, -1.0, 1.0) * 32767).astype(np.int16)
        yield (sr, audio_int16), chunk_status, None, None, progress_html
        time.sleep(0.05)

    # Final assembly
    if not all_chunk_audios or accumulated_audio is None or len(accumulated_audio) == 0:
        yield None, "❌ No audio generated", None, None, ""
        return

    # Apply remove_silence to FINAL result only, not individual chunks
    if remove_silence:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
            temp_int16 = (np.clip(accumulated_audio, -1.0, 1.0) * 32767).astype(np.int16)
            sf.write(temp_path, temp_int16, sr)
            remove_silence_for_generated_wav(temp_path)
            final_wave, _ = sf.read(temp_path)
            if final_wave.ndim > 1:
                final_wave = final_wave[:, 0]
            accumulated_audio = final_wave.astype(np.float32)
            os.unlink(temp_path)
        except Exception as e:
            logging.warning(f"[ChunkedTTS] Final silence removal failed: {e}")

    final_duration = len(accumulated_audio) / sr

    # Session complete - delete checkpoint
    checkpoint_mgr.delete(session_id)

    # Calculate timing stats
    total_generation_time = time.time() - progress.start_time
    realtime_factor = final_duration / total_generation_time if total_generation_time > 0 else 0
    avg_chunk_time = total_generation_time / total_chunks if total_chunks > 0 else 0

    # Apply normalization to final audio if enabled
    if normalize:
        accumulated_audio = normalize_audio(accumulated_audio, float(normalize_level))

    # Convert float32 accumulated audio to int16 ONCE for saving
    final_wave = (np.clip(accumulated_audio, -1.0, 1.0) * 32767).astype(np.int16)
    out_path = save_audio(final_wave, sr, output_format, actual_seed, target_sr=target_sample_rate, bitrate=bitrate, ogg_quality=ogg_quality)

    final_status = f"🎉 Generation complete!\n"
    final_status += f"📦 Total chunks: {total_chunks}\n"
    final_status += f"⏱️ Audio duration: {final_duration:.1f}s\n"
    final_status += f"⏱️ Generation time: {total_generation_time:.1f}s\n"
    final_status += f"⚡ Real-time factor: {realtime_factor:.2f}x\n"
    final_status += f"🎲 Seed: {actual_seed}\n"
    final_status += f"💾 Saved: {Path(out_path).name}"

    # Generate spectrogram from final accumulated audio
    try:
        final_spectrogram_path = save_spectrogram(final_wave, sr)
    except Exception as e:
        print(f"[Spectrogram] Failed to generate final spectrogram: {e}")
        final_spectrogram_path = None

    # Convert spectrogram to HTML with click-to-enlarge
    spectrogram_html = _spectrogram_to_html(final_spectrogram_path) if final_spectrogram_path else ""
    yield (sr, final_wave), final_status, out_path, spectrogram_html, ""
ROOT_DIR = os.environ.get("ROOT", os.getcwd())
CACHE_DIR = os.environ.get("CACHE", os.path.join(ROOT_DIR, "cache"))
VOICES_DIR = os.path.join(ROOT_DIR, "voices")

os.makedirs(VOICES_DIR, exist_ok=True)

import os


def download_and_extract(url):
    archive_path = os.path.join(CACHE_DIR, "voices_tmp.zip")

    if os.path.exists(archive_path):
        os.remove(archive_path)

    r = requests.get(url, stream=True)
    if r.status_code != 200:
        print(f"Error while downloading: {r.status_code}")
        invalidate_voice_cache()
    return gr.update(choices=get_voice_choices(), value="-NONE-")

    with open(archive_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(VOICES_DIR)

    os.remove(archive_path)

    return gr.update(choices=get_voice_choices(), value="-NONE-")



@functools.lru_cache(maxsize=1)
def _scan_voice_files() -> list:
    """Cached scan of voices directory. Avoids repeated disk I/O."""
    if not os.path.isdir(VOICES_DIR):
        return []
    valid_exts = (".wav", ".mp3", ".aac", ".m4a", ".m4b", ".ogg", ".flac", ".opus")
    return sorted([
        f for f in os.listdir(VOICES_DIR)
        if os.path.isfile(os.path.join(VOICES_DIR, f)) and f.lower().endswith(valid_exts)
    ])

def invalidate_voice_cache():
    """Call after downloading/updating voices."""
    _scan_voice_files.cache_clear()

def refresh_voices():
    """Refresh voice list with cache invalidation."""
    invalidate_voice_cache()
    return gr.update(choices=get_voice_choices(), value="-NONE-")


def get_voice_choices():
    """Get voice list — returns a FRESH list each call to avoid reactive reference issues."""
    return ["-NONE-"] + list(_scan_voice_files())

def clear_audio():
    return None

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
            "https://huggingface.co/datasets/LeeAeron/F5TTSx/resolve/main/models/ESpeech/ESpeech-TTS-1_RL-V1/espeech_tts_rlv1.pt"
        )
        vocab = safe_path(
            "models/ESpeech/ESpeech-TTS-1_RL-V1/vocab.txt",
            "https://huggingface.co/datasets/LeeAeron/F5TTSx/resolve/main/models/ESpeech/ESpeech-TTS-1_RL-V1/vocab.txt"
        )
        tts_model_choice = ("ESpeech-TTS-1_RL-V1", ckpt, vocab,
                            json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2,
                                            text_dim=512, conv_layers=4)))
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=True)

    elif new_choice == "ESpeech-TTS-1_RL-V2":
        ckpt = safe_path(
            "models/ESpeech/ESpeech-TTS-1_RL-V2/espeech_tts_rlv2.pt",
            "https://huggingface.co/datasets/LeeAeron/F5TTSx/resolve/main/models/ESpeech/ESpeech-TTS-1_RL-V2/espeech_tts_rlv2.pt"
        )
        vocab = safe_path(
            "models/ESpeech/ESpeech-TTS-1_RL-V2/vocab.txt",
            "https://huggingface.co/datasets/LeeAeron/F5TTSx/resolve/main/models/ESpeech/ESpeech-TTS-1_RL-V2/vocab.txt"
        )
        tts_model_choice = ("ESpeech-TTS-1_RL-V2", ckpt, vocab,
                            json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2,
                                            text_dim=512, conv_layers=4)))
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=True)

    elif new_choice == "ESpeech-TTS-1_SFT-95K":
        ckpt = safe_path(
            "models/ESpeech/ESpeech-TTS-1_SFT-95K/espeech_tts_95k.pt",
            "https://huggingface.co/datasets/LeeAeron/F5TTSx/resolve/main/models/ESpeech/ESpeech-TTS-1_SFT-95K/espeech_tts_95k.pt"
        )
        vocab = safe_path(
            "models/ESpeech/ESpeech-TTS-1_SFT-95K/vocab.txt",
            "https://huggingface.co/datasets/LeeAeron/F5TTSx/resolve/main/models/ESpeech/ESpeech-TTS-1_SFT-95K/vocab.txt"
        )
        tts_model_choice = ("ESpeech-TTS-1_SFT-95K", ckpt, vocab,
                            json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2,
                                            text_dim=512, conv_layers=4)))
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=True)

    elif new_choice == "ESpeech-TTS-1_SFT-256K":
        ckpt = safe_path(
            "models/ESpeech/ESpeech-TTS-1_SFT-256K/espeech_tts_256k.pt",
            "https://huggingface.co/datasets/LeeAeron/F5TTSx/resolve/main/models/ESpeech/ESpeech-TTS-1_SFT-256K/espeech_tts_256k.pt"
        )
        vocab = safe_path(
            "models/ESpeech/ESpeech-TTS-1_SFT-256K/vocab.txt",
            "https://huggingface.co/datasets/LeeAeron/F5TTSx/resolve/main/models/ESpeech/ESpeech-TTS-1_SFT-256K/vocab.txt"
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



def set_voice_file(selected_voice):
    if selected_voice and selected_voice != "-NONE-":
        file_path = os.path.join(VOICES_DIR, selected_voice)
        if os.path.isfile(file_path):
            return os.path.abspath(file_path)
        else:
            return None
    return None


# Basic TTS tab with smart chunking

# Load settings at startup
_settings = load_settings()

# Clean checkpoints on startup if setting is enabled
if _settings.get("clear_checkpoints_on_startup", False):
    cleanup_checkpoints_folder()


with gr.Blocks() as app_tts:
    with gr.Row():
        # LEFT COLUMN (2/5) — Settings, Voice, Reference, Advanced
        with gr.Column(scale=2, min_width=320):

            # Model & ASR
            choose_tts_model = gr.Dropdown(
                choices=[
                    ("🏠 F5-TTS v1 (EN/ZH)", "F5-TTS_v1"),
                    ("🇷🇺 Misha24-10 v2", "Misha24-10_v2"),
                    ("🇷🇺 Misha24-10 v4", "Misha24-10_v4"),
                    ("🇷🇺️ ESpeech Podcaster", "ESpeech-TTS-1_podcaster"),
                    ("🇷🇺 ESpeech RL-V1", "ESpeech-TTS-1_RL-V1"),
                    ("🇷🇺 ESpeech RL-V2", "ESpeech-TTS-1_RL-V2"),
                    ("🇷🇺 ESpeech SFT-95K", "ESpeech-TTS-1_SFT-95K"),
                    ("🇷🇺 ESpeech SFT-256K", "ESpeech-TTS-1_SFT-256K"),
                    ("⚙️ Custom", "Custom"),
                ],
                label="TTS Model",
                value="F5-TTS_v1",
            )

            # Custom model fields (hidden by default)
            custom_ckpt_path = gr.Dropdown(
                choices=[DEFAULT_TTS_MODEL_CFG[0]],
                value=DEFAULT_TTS_MODEL_CFG[0],
                allow_custom_value=True,
                label="Model path",
                visible=False,
            )
            custom_vocab_path = gr.Dropdown(
                choices=[DEFAULT_TTS_MODEL_CFG[1]],
                value=DEFAULT_TTS_MODEL_CFG[1],
                allow_custom_value=True,
                label="Vocab path",
                visible=False,
            )
            custom_model_cfg = gr.Dropdown(
                choices=[DEFAULT_TTS_MODEL_CFG[2]],
                value=DEFAULT_TTS_MODEL_CFG[2],
                allow_custom_value=True,
                label="Model config",
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

            # Voice selector

            voice_selector = gr.Dropdown(
                choices=get_voice_choices(),
                label="Voice",
                value="-NONE-",
                scale=4,
            )
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh List", elem_classes="btn-gray", scale=1)
                download_btn = gr.Button("⬇️ Download Voices", elem_classes="btn-gray", scale=1)

            ref_audio_input = gr.Audio(label="Reference Audio", type="filepath", visible=True)

            # Reference Text
            with gr.Accordion("Reference Text", open=False):
                ref_text_input = gr.Textbox(
                    label="Reference Text",
                    info="Leave blank to auto-transcribe. Upload file to override.",
                    lines=2,
                )
                ref_text_file = gr.File(
                    label="Reference Text from File (.txt)",
                    file_types=[".txt"],
                )

            # Output Format
            with gr.Accordion("🔊 Output Format", open=False):
                format_selector = gr.Dropdown(
                    choices=AUDIO_FORMATS,
                    value=_settings.get("output_format", "mp3"),
                    label="Output Format"
                )
                output_sample_rate = gr.Dropdown(
                    choices=OUTPUT_SAMPLE_RATES,
                    value=_settings.get("output_sample_rate", 48000),
                    label="Output Sample Rate (Hz)"
                )
                bitrate_selector = gr.Dropdown(
                    choices=BITRATE_OPTIONS,
                    value=_settings.get("bitrate", 320),
                    label="Bitrate (kbps)",
                    visible=(_settings.get("output_format", "wav") not in ("ogg", "opus")),
                )
                ogg_quality_selector = gr.Slider(
                    minimum=-1, maximum=10,
                    value=_settings.get("ogg_quality", 5), step=1,
                    label="OGG Quality (-1 to 10)",
                    info="Quality level for OGG Vorbis encoding. Higher = better quality, larger file.",
                    visible=(_settings.get("output_format", "wav") in ("ogg", "opus")),
                )
                normalize_audio_cb = gr.Checkbox(
                    label="Normalize Audio",
                    info="Normalize output to target RMS level",
                    value=_settings.get("normalize", True),
                )
                normalize_level = gr.Dropdown(
                    label="Normalization Level (dB)",
                    choices=NORMALIZATION_LEVELS,
                    value=_settings.get("normalize_level", -15),
                    interactive=True
                )

            # Format change handler for bitrate/quality visibility toggle
            def update_bitrate_visibility(fmt):
                fmt = fmt.lower() if fmt else "wav"
                is_ogg_opus = fmt in ("ogg", "opus")
                return (
                    gr.update(visible=not is_ogg_opus),   # bitrate_selector
                    gr.update(visible=is_ogg_opus),       # ogg_quality_selector
                )

            format_selector.change(
                fn=update_bitrate_visibility,
                inputs=[format_selector],
                outputs=[bitrate_selector, ogg_quality_selector],
                show_progress="hidden",
            )

            # Advanced Settings
            with gr.Accordion("🔧 Advanced Settings", open=True) as adv_settn:
                with gr.Row(elem_classes="seed-row"):
                    randomize_seed = gr.Checkbox(
                        label="Randomize Seed",
                        value=_settings.get("randomize_seed", True),
                        scale=3,
                    )
                    seed_input = gr.Number(
                        show_label=False,
                        value=_settings.get("seed", 0),
                        info="-1 = random seed",
                        precision=0,
                        scale=1,
                    )
                    random_seed_btn = gr.Button(
                        "🎲",
                        elem_classes="square-btn",
                        scale=0,
                        min_width=40
                    )
                remove_silence = gr.Checkbox(
                    label="Remove Silences",
                    value=_settings.get("remove_silence", False),
                )
                speed_slider = gr.Slider(
                    label="Speed", minimum=0.3, maximum=2.0,
                    value=_settings.get("speed", 1.0), step=0.1,
                )
                nfe_slider = gr.Slider(
                    label="NFE Steps", minimum=4, maximum=64,
                    value=_settings.get("nfe_step", 16), step=2,
                )
                cross_fade_duration_slider = gr.Slider(
                    label="Cross-Fade Duration (sec)", minimum=0.0, maximum=1.0,
                    value=_settings.get("cross_fade_duration", 0.0), step=0.01,
                )
                target_rms_input = gr.Slider(
                    label="Target RMS", minimum=0.01, maximum=1.0,
                    value=_settings.get("target_rms", 0.1), step=0.01,
                )
                cfg_strength_input = gr.Slider(
                    label="CFG Strength", minimum=0.1, maximum=5.0,
                    value=_settings.get("cfg_strength", 1.0), step=0.1,
                )
                use_accent_checkbox = gr.Checkbox(
                    label="RuAccent auto pronounce",
                    value=_settings.get("use_accent", False),
                )

            # Smart Chunking
            with gr.Accordion("📦 Smart Chunking", open=False):
                enable_chunking = gr.Checkbox(
                    label="Enable Smart Chunking",
                    value=_settings.get("enable_chunking", True),
                )
                chunking_mode = gr.Dropdown(
                    choices=[
                        ("Lines", "lines"),
                        ("Sentences (. ! ? …)", "sentences"),
                        ("Characters", "characters")
                    ],
                    value=_settings.get("chunking_mode", "lines"),
                    label="Chunking Mode",
                )
                with gr.Tabs() as chunking_tabs:
                    with gr.TabItem("Lines", id="lines"):
                        max_lines = gr.Slider(
                            minimum=1, maximum=20,
                            value=_settings.get("max_lines_per_chunk", 8), step=1,
                            label="Max Lines Per Chunk",
                        )
                    with gr.TabItem("Sentences", id="sentences"):
                        max_sentences = gr.Slider(
                            minimum=1, maximum=20,
                            value=_settings.get("max_sentences_per_chunk", 5), step=1,
                            label="Max Sentences Per Chunk",
                        )
                    with gr.TabItem("Characters", id="characters"):
                        max_chars = gr.Slider(
                            minimum=100, maximum=3000,
                            value=_settings.get("max_chars_per_chunk", 800), step=50,
                            label="Max Characters Per Chunk",
                        )
                crossfade_ms = gr.Slider(
                    minimum=0, maximum=200,
                    value=_settings.get("crossfade_ms", 200), step=10,
                    label="Crossfade Duration (ms)",
                )
                clean_checkpoints_cb = gr.Checkbox(
                    label="🗑️ Clean checkpoints on startup",
                    value=_settings.get("clear_checkpoints_on_startup", True),
                )



        # RIGHT COLUMN (3/5) — Generation, Output, Status
        with gr.Column(scale=3, min_width=400):

            # Text to Generate
            gen_text_input = gr.Textbox(
                label="Text to Generate",
                lines=10,
                max_lines=40,
                placeholder="Enter text to synthesize...",
            )
            # Action Buttons
            with gr.Row():
                clear_btn = gr.Button("🗑️ CLEAR", elem_classes="btn-gray", scale=1)
                paste_btn = gr.Button("📋 PASTE", elem_classes="btn-gray", scale=1)
                copy_btn = gr.Button("📄 COPY", elem_classes="btn-gray", scale=1)
                ruaccent_btn = gr.Button("🇷🇺 RUACCENT", elem_classes="btn-gray", scale=1)

            with gr.Accordion("Load prompt Text from File", open=False):
                gen_text_file = gr.File(
                    label="Load Text from File (.txt)",
                    file_types=[".txt"],
                )

            # Generate / Restart
            with gr.Row():
                generate_btn = gr.Button("▶ GENERATE", variant="primary", elem_classes="btn-generate", scale=3)
                restart_btn = gr.Button("🔄 RESTART UI", variant="stop", elem_classes="btn-restart", scale=2)

            # Progress Bar
            with gr.Row():
                progress_output = gr.HTML(value="", visible=True)

            # Outputs
            audio_output = gr.Audio(label="Generated Audio", autoplay=False)
            saved_output = gr.File(label="Saved Output")
            spectrogram_output = gr.HTML(label="Spectrogram")
            status_output = gr.Textbox(label="Status", lines=3, interactive=False)

    # Event handlers

    # TTS Model switch
    choose_tts_model.change(
        switch_tts_model,
        inputs=[choose_tts_model],
        outputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg, use_accent_checkbox],
        show_progress="hidden",
    )

    # Custom model fields
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

    # ASR
    choose_asr_model.change(
        set_asr_model,
        inputs=[choose_asr_model],
        outputs=[],
        show_progress="hidden",
    )

    # Voice selector
    refresh_btn.click(refresh_voices, inputs=None, outputs=voice_selector)

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

    # Reference audio clear
    ref_audio_input.clear(
        lambda: [None, None],
        None,
        [ref_text_input, ref_text_file],
    )

    # Voice selector clear audio when changed to -NONE-
    def clear_audio():
        return None

    # File uploads
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

    # Chunking mode tab switch
    chunking_mode.change(
        fn=lambda mode: gr.update(selected=mode),
        inputs=chunking_mode,
        outputs=chunking_tabs
    )

    # Random seed button handler
    def _random_seed():
        new_seed = random.randint(0, 2**32 - 1)
        return gr.update(value=new_seed)

    random_seed_btn.click(_random_seed, inputs=None, outputs=seed_input)

    # Text buttons
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
                alert("Error accessing  clipboard: " + err);
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

    # Restart engine
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

    # Collapse accordion workaround
    def collapse_accordion():
        return gr.Accordion(open=True)

    app_tts.load(
        fn=collapse_accordion,
        inputs=None,
        outputs=adv_settn,
    )

    # Generate wrapper

    def basic_tts_wrapper(
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
        output_sample_rate,
        bitrate,
        ogg_quality,
        normalize,
        normalize_level,
        enable_chunking,
        chunking_mode,
        max_lines,
        max_sentences,
        max_chars,
        crossfade_ms,
        clean_checkpoints,
    ):
        # Save settings before generation
        current_settings = {
            "speed": speed_slider,
            "nfe_step": nfe_slider,
            "cross_fade_duration": cross_fade_duration_slider,
            "target_rms": target_rms_input,
            "cfg_strength": cfg_strength_input,
            "use_accent": use_accent,
            "remove_silence": remove_silence,
            "seed": seed_input,
            "randomize_seed": randomize_seed,
            "output_format": output_format,
            "output_sample_rate": output_sample_rate,
            "bitrate": bitrate,
            "ogg_quality": ogg_quality,
            "normalize": normalize,
            "normalize_level": normalize_level,
            "enable_chunking": enable_chunking,
            "chunking_mode": chunking_mode,
            "max_lines_per_chunk": max_lines,
            "max_sentences_per_chunk": max_sentences,
            "max_chars_per_chunk": max_chars,
            "crossfade_ms": crossfade_ms,
            "clear_checkpoints_on_startup": clean_checkpoints,
        }
        save_settings(current_settings)

        if randomize_seed:
            seed_input = np.random.randint(0, 2**31 - 1)

        # Use the generator for chunked generation
        for audio_out, status_msg, saved_path, spectrogram_path, progress_html in generate_chunked_tts(
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            tts_model_choice,
            remove_silence,
            seed_input,
            cross_fade_duration_slider,
            nfe_slider,
            speed_slider,
            use_accent,
            target_rms_input,
            cfg_strength_input,
            output_format,
            normalize=normalize,
            normalize_level=normalize_level,
            enable_chunking=enable_chunking,
            chunking_mode=chunking_mode,
            max_lines_per_chunk=max_lines,
            max_sentences_per_chunk=max_sentences,
            max_chars_per_chunk=max_chars,
            crossfade_ms=crossfade_ms,
            target_sample_rate=output_sample_rate,
            bitrate=bitrate,
        ):
            yield audio_out, status_msg, saved_path, spectrogram_path, progress_html


    generate_btn.click(
        basic_tts_wrapper,
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
            output_sample_rate,
            bitrate_selector,
            ogg_quality_selector,
            normalize_audio_cb,
            normalize_level,
            enable_chunking,
            chunking_mode,
            max_lines,
            max_sentences,
            max_chars,
            crossfade_ms,
            clean_checkpoints_cb,
        ],
        outputs=[audio_output, status_output, saved_output, spectrogram_output, progress_output],
    )


def parse_speechtypes_text(gen_text: str) -> list:
    """Parse multi-style text into segments.

    Supports formats:
    - {Regular} Hello world
    - {"name": "Speaker1", "seed": 42, "speed": 1.2} Hello world
    """
    import json
    import re

    segments = []
    if not gen_text or not gen_text.strip():
        return segments

    # Pattern 1: {Regular} text
    # Pattern 2: {"name": "Speaker", "seed": -1, "speed": 1} text

    # Split by speech type markers
    # Match either {Name} or {"name": "...", ...}
    pattern = r'\{(?:[^}]+)\}'

    # Find all markers and their positions
    matches = list(re.finditer(pattern, gen_text))

    if not matches:
        # No markers found, treat as single Regular segment
        return [{"name": "Regular", "seed": -1, "speed": 1, "text": gen_text.strip()}]

    for i, match in enumerate(matches):
        marker = match.group(0)[1:-1]  # Remove { and }
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(gen_text)
        text = gen_text[start:end].strip()

        # Try to parse as JSON
        try:
            data = json.loads('{' + marker + '}')
            name = data.get("name", "Regular")
            seed = data.get("seed", -1)
            speed = data.get("speed", 1)
        except (json.JSONDecodeError, ValueError):
            # Simple name format: {Regular}
            name = marker.strip()
            seed = -1
            speed = 1

        if text:
            segments.append({"name": name, "seed": seed, "speed": speed, "text": text})

    return segments


with gr.Blocks() as app_multistyle:
    # multistyle generation
    gr.Markdown(
        """
    # Multiple Speech Generation
    """
    )

    with gr.Row():
        gr.Markdown(
            """
            **Example Input:** <br>
            {Regular} Hello, I'd like to order a sandwich please. <br>
            {Surprised} What do you mean you're out of bread? <br>
            """
        )

        gr.Markdown(
            """
            **Example Input 2:** <br>
            {"name": "Speaker1_Happy", "seed": -1, "speed": 1} Hello, I'd like to order a sandwich please. <br>
            {"name": "Speaker2_Regular", "seed": -1, "speed": 1} Sorry, we're out of bread.
            """
        )

    gr.Markdown(
        'Upload different audio clips for each speech type. The first speech type is mandatory. You can add additional speech types by clicking the "Add Speech Type" button.'
    )

    # MAIN LAYOUT: Left = Settings, Right = Multi-Speech Interface
    with gr.Row():
        # LEFT COLUMN (2/5) — Settings, Model, Format, Advanced, Chunking
        with gr.Column(scale=2, min_width=320):

            # Model & ASR
            choose_tts_model_multistyle = gr.Dropdown(
                choices=[
                    ("🏠 F5-TTS v1 (EN/ZH)", "F5-TTS_v1"),
                    ("🇷🇺 Misha24-10 v2", "Misha24-10_v2"),
                    ("🇷🇺 Misha24-10 v4", "Misha24-10_v4"),
                    ("🇷🇺 ESpeech Podcaster", "ESpeech-TTS-1_podcaster"),
                    ("🇷🇺 ESpeech RL-V1", "ESpeech-TTS-1_RL-V1"),
                    ("🇷🇺️ ESpeech RL-V2", "ESpeech-TTS-1_RL-V2"),
                    ("🇷🇺 ESpeech SFT-95K", "ESpeech-TTS-1_SFT-95K"),
                    ("🇷🇺️ ESpeech SFT-256K", "ESpeech-TTS-1_SFT-256K"),
                    ("⚙️ Custom", "Custom"),
                ],
                label="TTS Model",
                value="F5-TTS_v1",
            )

            # Custom model fields (hidden by default)
            custom_ckpt_path_multistyle = gr.Dropdown(
                choices=[DEFAULT_TTS_MODEL_CFG[0]],
                value=DEFAULT_TTS_MODEL_CFG[0],
                allow_custom_value=True,
                label="Model path",
                visible=False,
            )
            custom_vocab_path_multistyle = gr.Dropdown(
                choices=[DEFAULT_TTS_MODEL_CFG[1]],
                value=DEFAULT_TTS_MODEL_CFG[1],
                allow_custom_value=True,
                label="Vocab path",
                visible=False,
            )
            custom_model_cfg_multistyle = gr.Dropdown(
                choices=[DEFAULT_TTS_MODEL_CFG[2]],
                value=DEFAULT_TTS_MODEL_CFG[2],
                allow_custom_value=True,
                label="Model config",
                visible=False,
            )

            choose_asr_model_multistyle = gr.Dropdown(
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

            # Output Format
            with gr.Accordion("🔊 Output Format", open=False):
                format_selector_multistyle = gr.Dropdown(
                    choices=AUDIO_FORMATS,
                    value=_settings.get("output_format", "mp3"),
                    label="Output Format",
                )
                output_sample_rate_multistyle = gr.Dropdown(
                    choices=OUTPUT_SAMPLE_RATES,
                    value=_settings.get("output_sample_rate", 48000),
                    label="Output Sample Rate (Hz)",
                )
                bitrate_selector_multistyle = gr.Dropdown(
                    choices=BITRATE_OPTIONS,
                    value=_settings.get("bitrate", 320),
                    label="Bitrate (kbps)",
                    visible=(_settings.get("output_format", "wav") not in ("ogg", "opus")),
                )
                ogg_quality_selector_multistyle = gr.Slider(
                    minimum=-1, maximum=10,
                    value=_settings.get("ogg_quality", 5), step=1,
                    label="OGG Quality (-1 to 10)",
                    info="Quality level for OGG Vorbis encoding. Higher = better quality, larger file.",
                    visible=(_settings.get("output_format", "wav") in ("ogg", "opus")),
                )
                normalize_audio_cb_multistyle = gr.Checkbox(
                    label="Normalize Audio",
                    info="Normalize output to target RMS level",
                    value=_settings.get("normalize", True),
                )
                normalize_level_multistyle = gr.Dropdown(
                    label="Normalization Level (dB)",
                    choices=NORMALIZATION_LEVELS,
                    value=_settings.get("normalize_level", -15),
                    interactive=True,
                )

            # Format change handler for bitrate/quality visibility toggle
            def update_bitrate_visibility_multistyle(fmt):
                fmt = fmt.lower() if fmt else "wav"
                is_ogg_opus = fmt in ("ogg", "opus")
                return (
                    gr.update(visible=not is_ogg_opus),   # bitrate_selector
                    gr.update(visible=is_ogg_opus),       # ogg_quality_selector
                )

            format_selector_multistyle.change(
                fn=update_bitrate_visibility_multistyle,
                inputs=[format_selector_multistyle],
                outputs=[bitrate_selector_multistyle, ogg_quality_selector_multistyle],
                show_progress="hidden",
            )

            # Advanced Settings
            with gr.Accordion("🔧 Advanced Settings", open=False):
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

            # Smart Chunking
            with gr.Accordion("📦 Smart Chunking", open=False):
                enable_chunking_multistyle = gr.Checkbox(
                    label="Enable Smart Chunking",
                    value=_settings.get("enable_chunking", True),
                )
                chunking_mode_multistyle = gr.Dropdown(
                    choices=[
                        ("Lines", "lines"),
                        ("Sentences (. ! ? …)", "sentences"),
                        ("Characters", "characters")
                    ],
                    value=_settings.get("chunking_mode", "lines"),
                    label="Chunking Mode",
                )
                with gr.Tabs() as chunking_tabs_multistyle:
                    with gr.TabItem("Lines", id="lines"):
                        max_lines_multistyle = gr.Slider(
                            minimum=1, maximum=20,
                            value=_settings.get("max_lines_per_chunk", 8), step=1,
                            label="Max Lines Per Chunk",
                        )
                    with gr.TabItem("Sentences", id="sentences"):
                        max_sentences_multistyle = gr.Slider(
                            minimum=1, maximum=20,
                            value=_settings.get("max_sentences_per_chunk", 5), step=1,
                            label="Max Sentences Per Chunk",
                        )
                    with gr.TabItem("Characters", id="characters"):
                        max_chars_multistyle = gr.Slider(
                            minimum=100, maximum=3000,
                            value=_settings.get("max_chars_per_chunk", 800), step=50,
                            label="Max Characters Per Chunk",
                        )
                crossfade_ms_multistyle = gr.Slider(
                    minimum=0, maximum=200,
                    value=_settings.get("crossfade_ms", 200), step=10,
                    label="Crossfade Duration (ms)",
                )
                clean_checkpoints_cb_multistyle = gr.Checkbox(
                    label="🗑️ Clean checkpoints on startup",
                    value=_settings.get("clear_checkpoints_on_startup", True),
                )



        # RIGHT COLUMN (3/5) — Speech Types, Text Input, Generate, Outputs
        with gr.Column(scale=3):
            # Regular speech type (mandatory)
            with gr.Row(variant="compact") as regular_row:
                # Left narrow column
                with gr.Column(scale=1, min_width=160):
                    regular_name = gr.Textbox(value="Regular", label="Speech Name")
                    regular_insert = gr.Button("Insert Label", variant="secondary", elem_classes="btn-gray", scale=1)

            # Center speaker: voice + audio
            with gr.Column(scale=3):
                voice_selector = gr.Dropdown(
                    choices=get_voice_choices(),
                    label="Voice",
                    value="-NONE-",
                )
                refresh_btn = gr.Button("🔄 Refresh List", elem_classes="btn-gray", scale=1)
                regular_audio = gr.Audio(label="Regular Reference Audio", type="filepath")

            # Right column: text field + sliders
            with gr.Accordion("Reference Text", open=False):
                regular_ref_text = gr.Textbox(label="Reference Text (Regular)", lines=4)
                regular_ref_text_file = gr.File(
                    label="Load Reference Text from File (.txt)",
                    file_types=[".txt"]
                )
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
                    minimum=0.0, maximum=1.0, value=0.50, step=0.01
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

                # Right narrow column (kept for layout balance)
                with gr.Column(scale=1, min_width=160):
                    pass

            refresh_btn.click(
                refresh_voices,
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

            # Additional speech types (99 more)
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

            for i in range(max_speech_types - 1):
                with gr.Column(scale=3, visible=False) as row:
                    with gr.Row():
                        with gr.Column(scale=1, min_width=160):
                            name_input = gr.Textbox(label="Speech Name")
                            with gr.Row():
                                insert_btn = gr.Button("Insert Label", variant="secondary", elem_classes="btn-gray", scale=1)
                                delete_btn = gr.Button("Delete Type", variant="stop", elem_classes="btn-gray", scale=1)

                    with gr.Column(scale=3):
                        voice_selector = gr.Dropdown(
                            choices=get_voice_choices(),
                            label="Voice",
                            value="-NONE-",
                        )
                        refresh_btn = gr.Button("🔄 Refresh list", elem_classes="btn-gray", scale=1)
                        audio_input = gr.Audio(label="Reference Audio", type="filepath")

                    with gr.Accordion("Reference Text", open=False):
                        ref_text_input = gr.Textbox(label="Reference Text", lines=4)
                        ref_text_file_input = gr.File(
                            label="Load Reference Text from File (.txt)",
                            file_types=[".txt"]
                        )
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
                            minimum=0.0, maximum=1.0, value=0.50, step=0.01
                        )
                        nfe_input = gr.Slider(
                            label="NFE Steps",
                            minimum=4, maximum=64, value=16, step=2
                        )
                        cfg_input = gr.Slider(
                            label="CFG Strength",
                            minimum=0.1, maximum=5.0, value=2.0, step=0.1
                        )
                        accent_checkbox = gr.Checkbox(
                            label="RuAccent",
                            value=False,
                        )

                    with gr.Column(scale=1, min_width=160):
                        pass

                refresh_btn.click(
                    refresh_voices,
                    inputs=None,
                    outputs=voice_selector,
                )

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
                speech_type_accent_checkboxes.append(accent_checkbox)

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
            add_speech_type_btn = gr.Button("➕ Add Speech Type", elem_classes="btn-gray", scale=1)

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
            with gr.Accordion("Prompt from file", open=False):
                gen_text_file_multistyle = gr.File(label="Load Text to Generate from File (.txt)", file_types=[".txt"], scale=1)

            with gr.Row():
                clear_btn = gr.Button("🗑️ CLEAR", elem_classes="btn-gray", scale=1)
                paste_btn = gr.Button("📋 PASTE", elem_classes="btn-gray", scale=1)
                copy_btn = gr.Button("📄 COPY", elem_classes="btn-gray", scale=1)
                ruaccent_btn = gr.Button("🇷🇺 RUACCENT", elem_classes="btn-gray", scale=1)

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
                        gr.Warning("Please enter speech name before insert.")
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

            # Progress Bar for Multi-Speech
            with gr.Row():
                progress_output_multistyle = gr.HTML(value="", visible=True)

            # Generate button
            with gr.Row():
                generate_multistyle_btn = gr.Button("▶ GENERATE", variant="primary", elem_classes="btn-generate", scale=3)
                restart_btn = gr.Button("🔄 RESTART UI", variant="stop", elem_classes="btn-restart", scale=2)

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
            saved_output = gr.File(label="Saved Output")
            spectrogram_output_multistyle = gr.HTML(label="Spectrogram")
            cherrypick_interface_multistyle = gr.Textbox(
                label="Cherry-pick Interface",
                lines=10,
                max_lines=40,
                buttons=["copy"],
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

    # EVENT HANDLERS FOR LEFT COLUMN (outside the main Row)
    # TTS Model switch
    choose_tts_model_multistyle.change(
        switch_tts_model,
        inputs=[choose_tts_model_multistyle],
        outputs=[custom_ckpt_path_multistyle, custom_vocab_path_multistyle, custom_model_cfg_multistyle, regular_accent_checkbox],
        show_progress="hidden",
    )

    # Custom model fields
    custom_ckpt_path_multistyle.change(
        set_custom_model,
        inputs=[custom_ckpt_path_multistyle, custom_vocab_path_multistyle, custom_model_cfg_multistyle],
        show_progress="hidden",
    )
    custom_vocab_path_multistyle.change(
        set_custom_model,
        inputs=[custom_ckpt_path_multistyle, custom_vocab_path_multistyle, custom_model_cfg_multistyle],
        show_progress="hidden",
    )
    custom_model_cfg_multistyle.change(
        set_custom_model,
        inputs=[custom_ckpt_path_multistyle, custom_vocab_path_multistyle, custom_model_cfg_multistyle],
        show_progress="hidden",
    )

    # ASR
    choose_asr_model_multistyle.change(
        set_asr_model,
        inputs=[choose_asr_model_multistyle],
        outputs=[],
        show_progress="hidden",
    )

    # Chunking mode tab switch for multistyle
    chunking_mode_multistyle.change(
        fn=lambda mode: gr.update(selected=mode),
        inputs=chunking_mode_multistyle,
        outputs=chunking_tabs_multistyle
    )

    # Wrapper for multistyle generation with chunking support
    def multistyle_tts_wrapper(
        gen_text,
        # Speech type params
        *args
    ):
        # Unpack args according to input order
        idx = 0
        speech_type_names_list = args[idx : idx + max_speech_types]; idx += max_speech_types
        speech_type_audios_list = args[idx : idx + max_speech_types]; idx += max_speech_types
        speech_type_ref_texts_list = args[idx : idx + max_speech_types]; idx += max_speech_types
        remove_silence = args[idx]; idx += 1
        speech_type_crossfades_list = args[idx : idx + max_speech_types]; idx += max_speech_types
        speech_type_nfes_list = args[idx : idx + max_speech_types]; idx += max_speech_types
        speech_type_cfgs_list = args[idx : idx + max_speech_types]; idx += max_speech_types
        speech_type_accent_list = args[idx : idx + max_speech_types]; idx += max_speech_types
        output_format = args[idx]; idx += 1
        output_sample_rate = args[idx]; idx += 1
        bitrate = args[idx]; idx += 1
        ogg_quality = args[idx]; idx += 1
        normalize = args[idx]; idx += 1
        normalize_level = args[idx]; idx += 1
        enable_chunking = args[idx]; idx += 1
        chunking_mode = args[idx]; idx += 1
        max_lines = args[idx]; idx += 1
        max_sentences = args[idx]; idx += 1
        max_chars = args[idx]; idx += 1
        crossfade_ms = args[idx]; idx += 1
        clean_checkpoints = args[idx]; idx += 1

        # Save settings before generation
        current_settings = {
            "output_format": output_format,
            "output_sample_rate": output_sample_rate,
            "bitrate": bitrate,
            "ogg_quality": ogg_quality,
            "normalize": normalize,
            "normalize_level": normalize_level,
            "enable_chunking": enable_chunking,
            "chunking_mode": chunking_mode,
            "max_lines_per_chunk": max_lines,
            "max_sentences_per_chunk": max_sentences,
            "max_chars_per_chunk": max_chars,
            "crossfade_ms": crossfade_ms,
            "clear_checkpoints_on_startup": clean_checkpoints,
        }
        save_settings(current_settings)

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

        # Parsing text into segments
        segments = parse_speechtypes_text(gen_text)

        if not segments:
            yield None, "No segments found in text.", None, None, ""
            return

        # Collect all text to check if chunking is needed globally
        all_text = " ".join([seg["text"] for seg in segments])

        # Check if chunking should be used globally
        should_chunk = False
        if enable_chunking:
            if chunking_mode == "lines":
                total_lines = len([l for l in all_text.strip().split("\n") if l.strip()])
                should_chunk = total_lines > max_lines
            elif chunking_mode == "sentences":
                total_sentences = len(split_into_sentences(all_text))
                should_chunk = total_sentences > max_sentences
            elif chunking_mode == "characters":
                total_chars = len(all_text.strip())
                should_chunk = total_chars > max_chars

        # Initialize chunker if needed
        chunker = None
        crossfader = None
        if should_chunk:
            chunker = ScriptChunker(
                chunking_mode=chunking_mode,
                max_lines_per_chunk=max_lines,
                max_sentences_per_chunk=max_sentences,
                max_chars_per_chunk=max_chars,
            )
            crossfader = AudioCrossfader(fade_duration_ms=crossfade_ms, sample_rate=24000)

        generated_audio_segments = []
        inference_meta_data = ""
        sr = 24000

        # Progress tracking
        total_steps = len(segments)
        progress = ProgressTracker(total_steps)

        status_msg = f"📦 Multi-Speech: {total_steps} segments"
        if should_chunk:
            status_msg += f" (chunking enabled, mode: {chunking_mode})"
        progress_html = generate_progress_html(progress)
        yield None, status_msg, None, None, progress_html

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
                yield None, f"Missing reference audio for {current_type_name}", None, None, ""
                return

            ref_text = speech_types[current_type_name].get("ref_text", "")

            if seed_input == -1:
                seed_input = np.random.randint(0, 2**31 - 1)

            # If the checkbox is enabled, run the text through RuAccent
            if speech_type_accent_list[i]:
                text = preprocess_text(text)

            progress.start_chunk()
            chunk_status = f"🔄 Processing segment {i + 1}/{total_steps} ({current_type_name})..."
            progress_html = generate_progress_html(progress, i + 1)
            yield None, chunk_status, None, None, progress_html
            time.sleep(0.15)

            # Handle chunking for this segment
            segment_audios = []

            if should_chunk:
                # Chunk this segment's text
                chunks = chunker.parse_text(text)
                if not chunks:
                    # Fallback: treat whole text as single chunk
                    chunks = [ScriptChunk(text=text, start_idx=0, end_idx=len(text)-1, is_first=True, is_last=True)]

                for chunk_idx, chunk in enumerate(chunks):
                    try:
                        audio_out, spectrogram_path, ref_text_out, used_seed = infer(
                            ref_audio,
                            ref_text,
                            chunk.text,
                            tts_model_choice,
                            remove_silence=False,  # Never remove silence for individual chunks
                            seed=seed_input,
                            cross_fade_duration=speech_type_crossfades_list[i],
                            nfe_step=speech_type_nfes_list[i],
                            speed=speed,
                            cfg_strength=speech_type_cfgs_list[i],
                            show_info=print,
                        )
                        chunk_sr, chunk_waveform = audio_out
                        sr = chunk_sr

                        # Convert to float32 for processing
                        if chunk_waveform.dtype == np.int16:
                            chunk_float = chunk_waveform.astype(np.float32) / 32768.0
                        else:
                            chunk_float = chunk_waveform.astype(np.float32)

                        segment_audios.append(chunk_float)

                        # Update ref_text from first chunk
                        if chunk_idx == 0:
                            speech_types[current_type_name]["ref_text"] = ref_text_out

                    except Exception as e:
                        logging.error(f"[Segment {i+1} Chunk {chunk_idx+1}] Error: {e}")
                        continue

                # Crossfade all chunks of this segment
                if segment_audios:
                    segment_audio = segment_audios[0]
                    for next_chunk in segment_audios[1:]:
                        segment_audio = crossfader.apply_crossfade(segment_audio, next_chunk)
                else:
                    segment_audio = np.array([])
            else:
                # No chunking - single inference
                try:
                    audio_out, spectrogram_path, ref_text_out, used_seed = infer(
                        ref_audio,
                        ref_text,
                        text,
                        tts_model_choice,
                        remove_silence=False,  # Applied to final result only
                        seed=seed_input,
                        cross_fade_duration=speech_type_crossfades_list[i],
                        nfe_step=speech_type_nfes_list[i],
                        speed=speed,
                        cfg_strength=speech_type_cfgs_list[i],
                        show_info=print,
                    )
                    sr, segment_waveform = audio_out

                    if segment_waveform.dtype == np.int16:
                        segment_audio = segment_waveform.astype(np.float32) / 32768.0
                    else:
                        segment_audio = segment_waveform.astype(np.float32)

                    speech_types[current_type_name]["ref_text"] = ref_text_out

                except Exception as e:
                    logging.error(f"[Segment {i+1}] Error: {e}")
                    progress.finish_chunk()
                    continue

            if segment_audio is None or len(segment_audio) == 0:
                progress.finish_chunk()
                continue

            generated_audio_segments.append(segment_audio)
            inference_meta_data += json.dumps(dict(name=name, seed=used_seed, speed=speed)) + f" {text}\n"

            progress.finish_chunk()
            progress_html = generate_progress_html(progress, i + 1)
            segment_status = f"✅ Segment {i + 1}/{total_steps} ({current_type_name}): {len(segment_audio)/sr:.1f}s"
            yield None, segment_status, None, None, progress_html
            time.sleep(0.05)

        # Final assembly
        if not generated_audio_segments:
            yield None, "❌ No audio generated", None, None, ""
            return

        # Concatenate all segments
        final_audio_data = np.concatenate(generated_audio_segments)

        # Apply remove_silence to FINAL result only
        if remove_silence:
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    temp_path = f.name
                temp_int16 = (np.clip(final_audio_data, -1.0, 1.0) * 32767).astype(np.int16)
                sf.write(temp_path, temp_int16, sr)
                remove_silence_for_generated_wav(temp_path)
                final_wave, _ = sf.read(temp_path)
                if final_wave.ndim > 1:
                    final_wave = final_wave[:, 0]
                final_audio_data = final_wave.astype(np.float32)
                os.unlink(temp_path)
            except Exception as e:
                logging.warning(f"[MultiStyle] Final silence removal failed: {e}")

        # Apply normalization if enabled
        if normalize:
            final_audio_data = normalize_audio(final_audio_data, float(normalize_level))

        # Convert to int16 for saving
        final_wave_int16 = (np.clip(final_audio_data, -1.0, 1.0) * 32767).astype(np.int16)

        # Save with full ffmpeg options
        out_path = save_audio(
            final_wave_int16, 
            sr, 
            output_format, 
            seed_input, 
            target_sr=output_sample_rate, 
            bitrate=bitrate, 
            ogg_quality=ogg_quality
        )

        # Generate spectrogram from final audio
        try:
            final_spectrogram_path = save_spectrogram(final_wave_int16, sr)
            spectrogram_html = _spectrogram_to_html(final_spectrogram_path) if final_spectrogram_path else ""
        except Exception as e:
            print(f"[Spectrogram] Failed to generate final spectrogram: {e}")
            spectrogram_html = ""

        final_duration = len(final_audio_data) / sr
        total_generation_time = time.time() - progress.start_time
        realtime_factor = final_duration / total_generation_time if total_generation_time > 0 else 0

        final_status = f"🎉 Multi-Speech complete!\n"
        final_status += f"📦 Segments: {total_steps}\n"
        final_status += f"⏱️ Audio duration: {final_duration:.1f}s\n"
        final_status += f"⏱️ Generation time: {total_generation_time:.1f}s\n"
        final_status += f"⚡ Real-time factor: {realtime_factor:.2f}x\n"
        final_status += f"💾 Saved: {Path(out_path).name}"

        yield (sr, final_wave_int16), final_status, out_path, spectrogram_html, ""


    # generate multistyle btn - PORTED with new inputs and outputs
    generate_multistyle_btn.click(
        multistyle_tts_wrapper,
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
        + [format_selector_multistyle]
        + [output_sample_rate_multistyle]
        + [bitrate_selector_multistyle]
        + [ogg_quality_selector_multistyle]
        + [normalize_audio_cb_multistyle]
        + [normalize_level_multistyle]
        + [enable_chunking_multistyle]
        + [chunking_mode_multistyle]
        + [max_lines_multistyle]
        + [max_sentences_multistyle]
        + [max_chars_multistyle]
        + [crossfade_ms_multistyle]
        + [clean_checkpoints_cb_multistyle],
        outputs=[
            audio_output_multistyle,
            cherrypick_interface_multistyle,  # Using this for status since it's text-based
            saved_output,
            spectrogram_output_multistyle,
            progress_output_multistyle,
        ],
    )

    # Validation function to disable Generate button if speech types are missing
    def validate_speech_types(gen_text, regular_name, *args):
        speech_type_names_list = args
        speech_types_available = set()
        if regular_name:
            speech_types_available.add(regular_name)
        for name_input in speech_type_names_list:
            if name_input:
                speech_types_available.add(name_input)
        segments = parse_speechtypes_text(gen_text)
        speech_types_in_text = set(segment["name"] for segment in segments)
        missing_speech_types = speech_types_in_text - speech_types_available
        if missing_speech_types:
            return gr.update(interactive=False)
        else:
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
                    choices=get_voice_choices(),
                    label="Voice",
                    value="-NONE-",
                )
                # Update btn
                refresh_btn = gr.Button("🔄 Refresh list", elem_classes="btn-gray", scale=1)
                ref_audio_chat = gr.Audio(label="Reference Audio", type="filepath")
                with gr.Accordion("Reference Text", open=False):
                    ref_text_chat = gr.Textbox(
                        label="Reference Text",
                        info="Optional: Leave blank to auto-transcribe",
                        lines=2,
                        scale=3,
                    )
                    ref_text_file_chat = gr.File(
                        label="Load Reference Text from File (.txt)", file_types=[".txt"], scale=1
                    )
   
            with gr.Column():
                with gr.Accordion("Advanced Settings", open=False):
                    
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
                            value=0.50,
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
                refresh_voices,
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
F5-TTSx official [Git](https://github.com/LeeAeron/F5-TTSx)

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

* [LeeAeron](https://github.com/LeeAeron) — main code, repository, Hugginface space, features, installer/launcher, reference audios, dictionary support.
* [mrfakename](https://github.com/fakerybakery) — original [online demo](https://huggingface.co/spaces/mrfakename/E2-F5-TTS)
* [RootingInLoad](https://github.com/RootingInLoad) — chunk generation & podcast app exploration
* [jpgallegoar](https://github.com/jpgallegoar) — multiple speech-type generation & voice chat
* [Ebany Speech](https://huggingface.co/ESpeech) - additional russian language TTS models
"""
)



# Dark Theme
theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.Color(
        name="indigo",
        c50="#eef2ff",
        c100="#e0e7ff",
        c200="#c7d2fe",
        c300="#a5b4fc",
        c400="#818cf8",
        c500="#667eea",
        c600="#5b6fd6",
        c700="#4f5fbf",
        c800="#444fa8",
        c900="#3a3f91",
        c950="#2d2f6e",
    ),
    secondary_hue=gr.themes.colors.Color(
        name="purple",
        c50="#faf5ff",
        c100="#f3e8ff",
        c200="#e9d5ff",
        c300="#d8b4fe",
        c400="#c084fc",
        c500="#a855f7",
        c600="#9333ea",
        c700="#7e22ce",
        c800="#6b21a8",
        c900="#581c87",
        c950="#3b0764",
    ),
    neutral_hue=gr.themes.colors.Color(
        name="slate",
        c50="#f8fafc",
        c100="#f1f5f9",
        c200="#e2e8f0",
        c300="#cbd5e1",
        c400="#94a3b8",
        c500="#64748b",
        c600="#475569",
        c700="#334155",
        c800="#1e293b",
        c900="#0f172a",
        c950="#020617",
    ),
    font=["Inter", "Arial", "sans-serif"],
    font_mono=["ui-monospace", "Consolas", "monospace"],
)

css = """
/* === LAYOUT UTILITIES === */
.square-btn {width: 40px !important; min-width: 40px !important; padding: 0 !important; height: 40px !important;}
.voice-controls-row {align-items: center !important;}
.voice-controls-row button {height: 40px !important; margin-top: 24px !important;}
.generate-btn-row {margin-bottom: 10px !important;}
.seed-row {align-items: center !important; gap: 8px !important;}
.seed-row button {height: 40px !important; margin-top: 24px !important; min-width: 40px !important;}
.seed-row .form {margin-bottom: 0 !important;}
.chunking-row {align-items: center !important; gap: 8px !important;}
.chunking-row .form {margin-bottom: 0 !important;}

/* === SCROLLBAR === */
::-webkit-scrollbar {width: 8px; height: 8px;}
::-webkit-scrollbar-track {background: #0f172a;}
::-webkit-scrollbar-thumb {background: linear-gradient(#667eea, #764ba2); border-radius: 4px;}
::-webkit-scrollbar-thumb:hover {background: #667eea;}

/* === BUTTON COLORS === */
.btn-generate {
    background: linear-gradient(135deg, #10b981, #059669) !important;
    border-color: #059669 !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 15px !important;
}
.btn-generate:hover {
    background: linear-gradient(135deg, #34d399, #10b981) !important;
}

.btn-restart {
    background: linear-gradient(135deg, #ef4444, #dc2626) !important;
    border-color: #dc2626 !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}
.btn-restart:hover {
    background: linear-gradient(135deg, #f87171, #ef4444) !important;
}

.btn-gray {
    background: #334155 !important;
    border-color: #475569 !important;
    color: #e2e8f0 !important;
    font-size: 11px !important;
    padding: 4px 8px !important;
}
.btn-gray:hover {
    background: #475569 !important;
}

/* === PROGRESS BAR ISOLATION === */
.progress-wrap {
    min-height: 90px;
    margin: 4px 0 12px 0;
}

/* === PROGRESS BAR ISOLATION === */
.progress-row {
    min-height: 100px !important;
    max-height: 130px !important;
    overflow: hidden !important;
    margin: 8px 0 !important;
}
.progress-container {
    min-height: 100px !important;
    max-height: 130px !important;
    overflow: hidden !important;
}
.progress-container > div {
    height: auto !important;
    max-height: 130px !important;
}

/* === COMPACT LEFT COLUMN === */
.left-col .form {
    margin-bottom: 6px !important;
}
.left-col .wrap {
    margin-bottom: 4px !important;
}

/* === SPECTROGRAM FULLSCREEN FIX === */
.image-fullscreen {
    width: 100vw !important;
    height: 100vh !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    background: #000 !important;
}
.image-fullscreen img {
    max-width: 95vw !important;
    max-height: 95vh !important;
    object-fit: contain !important;
    width: auto !important;
    height: auto !important;
}
.wide-col .wrap {
    width: 100% !important;
}
.wide-col .gradio-slider,
.wide-col .gradio-textbox,
.wide-col .gradio-dropdown,
.wide-col .gradio-file {
    width: 100% !important;
}
"""


with gr.Blocks(title="F5-TTSx") as app:
    gr.Markdown(
        "<h1 style='text-align:center; margin-bottom:8px; font-size:28px;'>🎙️ F5-TTSx Portable</h1>",
    )
    
    # GPU Status
    if torch.cuda.is_available():
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        gr.Markdown(f'<div style="text-align:center;padding:10px;border-radius:5px;">🟢 GPU: {torch.cuda.get_device_name(0)} VRAM: {vram_total:.1f}GB</div>')
    else:
        gr.Markdown('<div style="text-align:center;padding:10px;border-radius:5px;">⚪ CPU Mode</div>')
    
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
                "https://huggingface.co/datasets/LeeAeron/F5TTSx/resolve/main/models/ESpeech/ESpeech-TTS-1_RL-V1/espeech_tts_rlv1.pt"
            )
            vocab = safe_path(
                "models/ESpeech/ESpeech-TTS-1_RL-V1/vocab.txt",
                "https://huggingface.co/datasets/LeeAeron/F5TTSx/resolve/main/models/ESpeech/ESpeech-TTS-1_RL-V1/vocab.txt"
            )
            tts_model_choice = ("ESpeech-TTS-1_RL-V1", ckpt, vocab,
                                json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2,
                                                text_dim=512, conv_layers=4)))
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=True)

        elif new_choice == "ESpeech-TTS-1_RL-V2":
            ckpt = safe_path(
                "models/ESpeech/ESpeech-TTS-1_RL-V2/espeech_tts_rlv2.pt",
                "https://huggingface.co/datasets/LeeAeron/F5TTSx/resolve/main/models/ESpeech/ESpeech-TTS-1_RL-V2/espeech_tts_rlv2.pt"
            )
            vocab = safe_path(
                "models/ESpeech/ESpeech-TTS-1_RL-V2/vocab.txt",
                "https://huggingface.co/datasets/LeeAeron/F5TTSx/resolve/main/models/ESpeech/ESpeech-TTS-1_RL-V2/vocab.txt"
            )
            tts_model_choice = ("ESpeech-TTS-1_RL-V2", ckpt, vocab,
                                json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2,
                                                text_dim=512, conv_layers=4)))
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=True)

        elif new_choice == "ESpeech-TTS-1_SFT-95K":
            ckpt = safe_path(
                "models/ESpeech/ESpeech-TTS-1_SFT-95K/espeech_tts_95k.pt",
                "https://huggingface.co/datasets/LeeAeron/F5TTSx/resolve/main/models/ESpeech/ESpeech-TTS-1_SFT-95K/espeech_tts_95k.pt"
            )
            vocab = safe_path(
                "models/ESpeech/ESpeech-TTS-1_SFT-95K/vocab.txt",
                "https://huggingface.co/datasets/LeeAeron/F5TTSx/resolve/main/models/ESpeech/ESpeech-TTS-1_SFT-95K/vocab.txt"
            )
            tts_model_choice = ("ESpeech-TTS-1_SFT-95K", ckpt, vocab,
                                json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2,
                                                text_dim=512, conv_layers=4)))
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=True)

        elif new_choice == "ESpeech-TTS-1_SFT-256K":
            ckpt = safe_path(
                "models/ESpeech/ESpeech-TTS-1_SFT-256K/espeech_tts_256k.pt",
                "https://huggingface.co/datasets/LeeAeron/F5TTSx/resolve/main/models/ESpeech/ESpeech-TTS-1_SFT-256K/espeech_tts_256k.pt"
            )
            vocab = safe_path(
                "models/ESpeech/ESpeech-TTS-1_SFT-256K/vocab.txt",
                "https://huggingface.co/datasets/LeeAeron/F5TTSx/resolve/main/models/ESpeech/ESpeech-TTS-1_SFT-256K/vocab.txt"
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

    # tabs
    gr.TabbedInterface(
        [app_tts, app_multistyle, app_chat, app_additional],
        ["Basic-TTS", "Multi-Speech", "Voice-Chat", "Additional"],
    )

    gr.Markdown("""
        ### 💡 Tips for Use:
        - **If you're having issues, use reference audio clipping it to 12s with  ✂  in the bottom right corner (otherwise might have non-optimal auto-trimmed result).
        - **Reference text will be automatically transcribed with Whisper if not provided. For best results, keep your reference clips short (<12s). Ensure the audio is fully uploaded before generating.
        - **Smart Chunking**: Enable for scripts >1000 words to prevent voice degradation
        - **Chunking Modes**: 
          - *Lines*: Original behavior, splits by line count
          - *Sentences*: Respects sentence boundaries (. ! ? …), best for natural speech
          - *Characters*: Simple character count, most predictable
        - **Clean Checkpoints**: Enable "Clean checkpoints folder on startup" to auto-delete old chunk files
        - **Crossfade**: 50ms is good for most cases; increase for smoother transitions

        ### 💡 Chunking:
        - **For short texts (< 1000 words): chunking is not enabled automatically - it works as before
        - **For long texts: just paste the text, chunking will turn on automatically
        - **If your voice is "crashing": reduce chunk size (lines/sentences/chars)
        - **If crossfade is audible: increase "Crossfade Duration" to 200ms
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
        theme=theme,
        css=css,
    )


if __name__ == "__main__":
    if not USING_SPACES:
        main()
    else:
        app.queue().launch(theme=theme, css=css)
