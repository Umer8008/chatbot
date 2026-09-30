"""
config.py — Application Configuration

This module is the single source of truth for all configurable values.
It reads from environment variables (loaded from .env) and provides
typed constants that the rest of the application imports.

Why a dedicated config module?
- Prevents hard-coded paths scattered across multiple files
- Makes configuration changes easy (one place to edit)
- Prepares the project for environment-based deployment
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from the project root.
# python-dotenv reads key=value pairs and injects them into os.environ.
load_dotenv()

# ─────────────────────────────────────────────
# Project Paths
# ─────────────────────────────────────────────

# Resolve the project root regardless of where the script is run from.
# __file__ is config.py → parent is src/ → parent is project root.
PROJECT_ROOT = Path(__file__).parent.parent

# Default model directory.
MODELS_DIR = PROJECT_ROOT / "models" / "mistral"

# The model filename. Override via .env: MODEL_FILENAME=your-model.gguf
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

# Full path to the GGUF model file.
MODEL_PATH = str(MODELS_DIR / MODEL_FILENAME)

# ─────────────────────────────────────────────
# Model Identity (for UI display)
# ─────────────────────────────────────────────

MODEL_NAME = os.getenv("MODEL_NAME", "TinyLlama 1.1B Chat v1.0")
MODEL_QUANTIZATION = os.getenv("MODEL_QUANTIZATION", "Q4_K_M")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0")

# ─────────────────────────────────────────────
# LLM Generation Parameters (Defaults)
# ─────────────────────────────────────────────

# Temperature controls randomness.
# 0.0 = fully deterministic, 1.5 = very creative/random.
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))

# Maximum number of new tokens the model generates per response.
# Higher = longer responses but more RAM and slower generation.
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "512"))

# Context window size in tokens (input + output combined).
# TinyLlama 1.1B supports up to 2048 tokens.
MODEL_CONTEXT_SIZE = int(os.getenv("MODEL_CONTEXT_SIZE", "2048"))

# ─────────────────────────────────────────────
# CPU Threading
# ─────────────────────────────────────────────

# Number of CPU threads for inference.
# For i7-1355U (10 cores / 12 threads), 8 threads is a safe default
# that leaves headroom for the OS and Streamlit.
N_THREADS = int(os.getenv("N_THREADS", "8"))

# Number of GPU layers to offload.
# Set to 0 since we have no dedicated GPU (Intel UHD is integrated).
# Setting this > 0 without a supported GPU will cause a crash.
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "0"))

# ─────────────────────────────────────────────
# Application Settings
# ─────────────────────────────────────────────

# Application display name
APP_TITLE = "Local AI Chatbot"
APP_DESCRIPTION = "Conversational AI powered by TinyLlama 1.1B · Running 100% locally"

# Default system prompt — defines the assistant's behavior and persona.
# Users can override this in the Streamlit sidebar at runtime.
DEFAULT_SYSTEM_PROMPT = """You are a helpful, knowledgeable, and honest AI assistant running entirely on the user's local machine. No data leaves their computer.

Your behavior guidelines:
- Answer questions clearly and accurately
- If you are uncertain about something, say so honestly rather than guessing
- Keep responses concise but complete
- Be conversational and friendly
- Remember the context of the conversation"""

# Maximum conversation history turns to keep in context.
# Each "turn" = 1 user message + 1 assistant message.
# At 4096 context size, keeping the last 10 turns is a safe limit.
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "10"))
