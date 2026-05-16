"""
audio_processing.py — DSP utility for clinical voice sample standardization.

Converts any uploaded/recorded web audio byte stream into a clean,
analysis-ready PCM 16-bit WAV file at 44.1 kHz mono.

Usage:
    from src.audio_processing import process_audio_signal

    wav_path = process_audio_signal(uploaded_file)
    # wav_path is now safe to pass to parselmouth / librosa extractors
"""
import os
import io
import uuid
import tempfile
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd


# ── Constants ─────────────────────────────────────────────────────────────
TARGET_SAMPLE_RATE = 44100       # Clinical standard: 44.1 kHz
TARGET_CHANNELS    = 1           # Mono
TARGET_SUBTYPE     = "PCM_16"    # 16-bit signed integer PCM
MIN_DURATION_SEC   = 3.0         # Minimum usable phonation length


def process_audio_signal(audio_input_data) -> str:
    """
    Standardize a raw web audio byte stream into a clean WAV file.

    Pipeline:
        1. Validate & decode the audio bytes (any format soundfile supports)
        2. Down-mix to mono if stereo/multi-channel
        3. Resample to 44.1 kHz using polyphase anti-aliasing filter
        4. Enforce minimum 3-second duration
        5. Write to a temporary PCM-16 WAV file on disk

    Parameters
    ----------
    audio_input_data : st.runtime.uploaded_file_manager.UploadedFile | bytes | io.BytesIO
        The raw audio object from st.audio_input(), a file upload, or raw bytes.

    Returns
    -------
    str
        Absolute path to the processed temporary .wav file.

    Raises
    ------
    ValueError
        If the audio is invalid, unreadable, or shorter than 3 seconds.
    """

    # ── Step 0: Normalize input to a seekable byte buffer ─────────────
    byte_buffer = _normalize_to_buffer(audio_input_data)

    # ── Step 1: Decode audio bytes into a NumPy signal array ──────────
    signal, original_sr = _decode_audio(byte_buffer)

    # ── Step 2: Down-mix to mono ──────────────────────────────────────
    signal = _to_mono(signal)

    # ── Step 3: Resample to 44.1 kHz ─────────────────────────────────
    signal = _resample(signal, original_sr, TARGET_SAMPLE_RATE)

    # ── Step 4: Validate minimum duration ─────────────────────────────
    duration_sec = len(signal) / TARGET_SAMPLE_RATE
    if duration_sec < MIN_DURATION_SEC:
        raise ValueError(
            f"Recording too short ({duration_sec:.1f}s). "
            f"Please record at least {MIN_DURATION_SEC:.0f} seconds of "
            f"sustained \"Ahhh\" for an accurate clinical analysis."
        )

    # ── Step 5: Write standardized WAV to a temp file ─────────────────
    output_path = _write_temp_wav(signal, TARGET_SAMPLE_RATE)

    return output_path


# ══════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ══════════════════════════════════════════════════════════════════════════

def _normalize_to_buffer(audio_input_data) -> io.BytesIO:
    """Convert any supported input type into a seekable BytesIO buffer."""
    if isinstance(audio_input_data, io.BytesIO):
        audio_input_data.seek(0)
        return audio_input_data

    if isinstance(audio_input_data, (bytes, bytearray)):
        return io.BytesIO(audio_input_data)

    # Streamlit UploadedFile has a .read() method
    if hasattr(audio_input_data, "read"):
        raw = audio_input_data.read()
        if hasattr(audio_input_data, "seek"):
            audio_input_data.seek(0)      # Reset for potential re-use
        return io.BytesIO(raw)

    raise ValueError(
        "Unsupported audio input. Please use the microphone recorder "
        "or upload a standard audio file (WAV, FLAC, OGG)."
    )


def _decode_audio(byte_buffer: io.BytesIO):
    """
    Decode audio bytes into a float64 NumPy array + sample rate.

    soundfile supports WAV, FLAC, OGG/Vorbis, and other libsndfile formats
    natively without requiring FFmpeg.
    """
    try:
        signal, sr = sf.read(byte_buffer, dtype="float64", always_2d=True)
    except Exception:
        raise ValueError(
            "Could not read the audio file. Please ensure it is a valid "
            "WAV, FLAC, or OGG recording. Compressed formats like MP3 "
            "are not supported — re-record using the built-in microphone."
        )

    if signal.size == 0:
        raise ValueError(
            "The audio file appears to be empty. "
            "Please try recording again."
        )

    return signal, sr


def _to_mono(signal: np.ndarray) -> np.ndarray:
    """
    Down-mix multi-channel audio to mono by averaging across channels.
    Input shape: (n_samples, n_channels) — always_2d from sf.read.
    """
    if signal.ndim == 2 and signal.shape[1] > 1:
        signal = np.mean(signal, axis=1)
    else:
        signal = signal.ravel()
    return signal


def _resample(signal: np.ndarray, sr_orig: int, sr_target: int) -> np.ndarray:
    """
    Polyphase resampling with anti-aliasing.

    Uses scipy.signal.resample_poly for efficient integer-ratio resampling.
    Falls back to a direct ratio if the GCD-reduced fraction is too large.
    """
    if sr_orig == sr_target:
        return signal

    # Compute the simplest integer ratio  up / down
    divisor = gcd(sr_target, sr_orig)
    up   = sr_target // divisor
    down = sr_orig   // divisor

    # Guard against pathological ratios (e.g., 44100/48000 = 147/160)
    # resample_poly handles these fine, but cap at reasonable values
    if up > 1000 or down > 1000:
        # Fall back to brute-force resampling for weird sample rates
        target_len = int(len(signal) * sr_target / sr_orig)
        return np.interp(
            np.linspace(0, len(signal) - 1, target_len),
            np.arange(len(signal)),
            signal,
        )

    return resample_poly(signal, up, down).astype(np.float64)


def _write_temp_wav(signal: np.ndarray, sr: int) -> str:
    """
    Write the processed signal to a uniquely-named temporary WAV file.

    The file is placed in the OS temp directory and is NOT auto-deleted
    so downstream consumers (parselmouth, librosa) can open it by path.
    Callers are responsible for cleanup after feature extraction.
    """
    tmp_dir  = tempfile.gettempdir()
    filename = f"pd_voice_{uuid.uuid4().hex[:12]}.wav"
    filepath = os.path.join(tmp_dir, filename)

    sf.write(filepath, signal, sr, subtype=TARGET_SUBTYPE)

    return filepath


# ══════════════════════════════════════════════════════════════════════════
#  Custom Exception
# ══════════════════════════════════════════════════════════════════════════

class InvalidAudioError(Exception):
    """
    Raised when a WAV file fails pre-pipeline quality checks.

    Designed to be caught by Streamlit frontends to display a red
    st.error() banner with a patient-friendly description.
    """
    pass


# ══════════════════════════════════════════════════════════════════════════
#  File-path-based QA validation gate
# ══════════════════════════════════════════════════════════════════════════

def validate_audio_properties(file_path: str) -> str:
    """
    Inspect and sanitize a WAV file on disk before feature extraction.

    Validation pipeline:
        1. Readability — open the file; reject if corrupt or empty
        2. Channel check — stereo → mono downmix (average of L+R)
        3. Sample rate — resample to exactly 44,100 Hz if different
        4. Bit depth — re-encode as 16-bit PCM if necessary

    If any correction is made, a new standardized file is written
    and its path returned.  The original file is left untouched.

    Parameters
    ----------
    file_path : str
        Absolute path to a .wav file on disk.

    Returns
    -------
    str
        Path to the verified (or newly created standardized) WAV file.
        If the file already meets all criteria, the original path is returned.

    Raises
    ------
    InvalidAudioError
        If the file is missing, unreadable, corrupt, or completely empty.
    """

    # ── Guard: file exists ────────────────────────────────────────────
    if not os.path.exists(file_path):
        raise InvalidAudioError(
            f"Audio file not found: {file_path}. "
            "Please re-upload or re-record your voice sample."
        )

    # ── Step 1: Read and validate ─────────────────────────────────────
    try:
        info = sf.info(file_path)
    except Exception as exc:
        raise InvalidAudioError(
            "The uploaded file is not a valid audio file or is corrupted. "
            "Please upload an uncompressed .wav recording."
        ) from exc

    try:
        signal, sr = sf.read(file_path, dtype="float64", always_2d=True)
    except Exception as exc:
        raise InvalidAudioError(
            "Failed to decode the audio data. The file may be truncated "
            "or encoded in an unsupported codec. Please use standard "
            "uncompressed PCM WAV format."
        ) from exc

    if signal.size == 0 or signal.shape[0] == 0:
        raise InvalidAudioError(
            "The audio file is empty (0 samples). "
            "Please upload a recording that contains audible speech."
        )

    # ── Track whether corrections are needed ──────────────────────────
    needs_rewrite = False

    # ── Step 2: Channel validation ────────────────────────────────────
    n_channels = signal.shape[1]
    if n_channels > 1:
        # Downmix to mono by averaging channels
        # This prevents spatial phase cancellation artefacts
        signal = np.mean(signal, axis=1, keepdims=True)
        needs_rewrite = True
    # Flatten to 1-D for downstream processing
    signal_mono = signal.ravel()

    # ── Step 3: Sample rate standardization ───────────────────────────
    if sr != TARGET_SAMPLE_RATE:
        signal_mono = _resample(signal_mono, sr, TARGET_SAMPLE_RATE)
        needs_rewrite = True

    # ── Step 4: Bit depth verification ────────────────────────────────
    # soundfile reports subtype as e.g. 'PCM_16', 'PCM_24', 'FLOAT'
    if info.subtype != TARGET_SUBTYPE:
        needs_rewrite = True

    # ── Write corrected file if any property was out of spec ──────────
    if needs_rewrite:
        corrected_path = _write_temp_wav(signal_mono, TARGET_SAMPLE_RATE)
        return corrected_path

    # All checks passed — original file is clean
    return file_path
