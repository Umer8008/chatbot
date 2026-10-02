"""
utils.py — Helper Functions and Validation


This module contains utility functions used across the application.
Keeping these here prevents code duplication and keeps other modules focused.
"""

import os
import platform
import re
from pathlib import Path
from typing import Optional, Tuple

from src.config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    MODEL_CONTEXT_SIZE,
    MODEL_PATH,
    MODEL_QUANTIZATION,
)


# ─────────────────────────────────────────────
# Model Validation
# ─────────────────────────────────────────────

def check_model_exists() -> Tuple[bool, str]:
    """
    Check whether the GGUF model file exists at the configured path.

    Returns:
        Tuple of (exists: bool, message: str)
        - If exists=True:  message contains the file size
        - If exists=False: message contains download instructions

    Example:
        ok, msg = check_model_exists()
        if not ok:
            st.error(msg)
            st.stop()
    """
    path = Path(MODEL_PATH)

    if path.exists() and path.is_file():
        size_bytes = path.stat().st_size
        size_str = format_file_size(size_bytes)
        return True, f"Model loaded: {path.name} ({size_str})"

    return False, (
        f"❌ Model file not found at:\n`{MODEL_PATH}`\n\n"
        f"**How to fix:**\n"
        f"1. Download the model from Hugging Face:\n"
        f"   `mistral-7b-instruct-v0.2.Q2_K.gguf`\n"
        f"   URL: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF\n"
        f"2. Place it in: `models/mistral/`\n"
        f"3. Restart the application."
    )


def estimate_ram_warning() -> Optional[str]:
    """
    Estimate RAM usage for the current quantization and warn if it may be tight.

    Returns:
        A warning string if RAM may be insufficient, None if likely OK.
    """
    # Approximate RAM requirements per quantization (in GB)
    ram_requirements = {
        "Q2_K":   3.5,
        "Q3_K_S": 4.0,
        "Q3_K_M": 4.5,
        "Q4_K_M": 5.5,
        "Q5_K_M": 6.5,
        "Q6_K":   7.5,
        "Q8_0":   9.0,
    }

    required_gb = ram_requirements.get(MODEL_QUANTIZATION, 4.0)

    try:
        import psutil  # optional dependency
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        if available_gb < required_gb:
            return (
                f"⚠️ Low RAM Warning: {available_gb:.1f} GB available, "
                f"~{required_gb:.1f} GB recommended for {MODEL_QUANTIZATION}. "
                f"Expect slow performance."
            )
    except ImportError:
        # psutil not installed — skip the check
        pass

    return None


# ─────────────────────────────────────────────
# Parameter Validation
# ─────────────────────────────────────────────

def validate_temperature(value: float) -> Tuple[bool, str]:
    """
    Validate a temperature value.

    Valid range: 0.0 to 2.0
    Recommended range: 0.1 to 1.2 for chat

    Args:
        value: Temperature value to validate

    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    if not isinstance(value, (int, float)):
        return False, "Temperature must be a number."
    if value < 0.0:
        return False, "Temperature cannot be negative."
    if value > 2.0:
        return False, "Temperature above 2.0 is not recommended."
    return True, "OK"


def validate_max_tokens(value: int) -> Tuple[bool, str]:
    """
    Validate a max_tokens value against the model context size.

    Args:
        value: Max tokens value to validate

    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    if not isinstance(value, int):
        return False, "Max tokens must be an integer."
    if value < 1:
        return False, "Max tokens must be at least 1."
    if value > MODEL_CONTEXT_SIZE:
        return False, (
            f"Max tokens ({value}) cannot exceed context size ({MODEL_CONTEXT_SIZE})."
        )
    return True, "OK"


def validate_user_input(text: str) -> Tuple[bool, str]:
    """
    Validate the user's chat input before sending to the model.

    Args:
        text: The raw user input string

    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    if not text or not text.strip():
        return False, "Please enter a message before sending."
    if len(text.strip()) > 4000:
        return False, "Message too long. Please keep it under 4000 characters."
    return True, "OK"


# ─────────────────────────────────────────────
# Formatting Helpers
# ─────────────────────────────────────────────

def format_file_size(size_bytes: int) -> str:
    """
    Convert a byte count to a human-readable string.

    Examples:
        format_file_size(2_700_000_000) → "2.7 GB"
        format_file_size(512_000)       → "500.0 KB"

    Args:
        size_bytes: File size in bytes

    Returns:
        Human-readable string with appropriate unit
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def get_system_info() -> dict:
    """
    Collect basic system information for display in the sidebar.

    Returns:
        Dict with OS, CPU, Python version info.
    """
    return {
        "os": platform.system() + " " + platform.release(),
        "python": platform.python_version(),
        "cpu_cores": os.cpu_count() or "N/A",
        "architecture": platform.machine(),
    }


def clean_model_response(text: str) -> str:
    """
    Clean up any residual formatting artifacts from the model output.

    Some models occasionally emit stop tokens or formatting characters
    in their output. This function strips them for clean display.

    Args:
        text: Raw model output string

    Returns:
        Cleaned string
    """
    # Remove common Mistral Instruct stop tokens if they leak through
    artifacts = ["</s>", "[INST]", "[/INST]", "<s>"]
    for artifact in artifacts:
        text = text.replace(artifact, "")

    # Normalize excessive newlines (more than 2 in a row → 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
