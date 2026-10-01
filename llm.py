"""
llm.py — Local LLM Loader

This module is responsible for loading and configuring the local Mistral model.

Key design decisions explained:

WHY GGUF?
  GGUF (GPT-Generated Unified Format) is the standard format for running
  quantized LLMs locally. It stores model weights in a compact binary format
  that llama.cpp can load directly without needing PyTorch or CUDA.

WHY llama.cpp?
  llama.cpp is a C++ inference engine optimized for running LLMs on CPU.
  It implements highly optimized matrix operations (AVX2/AVX512 on x86)
  that make local inference practical without a GPU.

WHY llama-cpp-python?
  It's Python bindings for llama.cpp — gives us a Python API to control
  the C++ engine. LangChain's LlamaCpp integration wraps this library.

WHY @st.cache_resource?
  Loading a 3+ GB model takes 10–30 seconds. Without caching, every Streamlit
  interaction would reload the model. cache_resource is Streamlit's mechanism
  for caching objects that are expensive to create and should be shared across
  all user sessions (like database connections or ML models).

WHY NOT reload per message?
  Model loading involves reading gigabytes from disk and allocating memory.
  The model weights don't change between messages, so reloading is pure waste.
"""

import logging
from typing import Optional

import streamlit as st
from langchain_community.llms import LlamaCpp

from src.config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    MODEL_CONTEXT_SIZE,
    MODEL_PATH,
    N_GPU_LAYERS,
    N_THREADS,
)

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def load_llm(
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Optional[LlamaCpp]:
    """
    Load the local Mistral GGUF model via llama.cpp.

    This function is decorated with @st.cache_resource, which means Streamlit
    calls it only ONCE and reuses the result for every subsequent interaction.
    The model stays loaded in RAM for the lifetime of the Streamlit process.

    Args:
        temperature: Sampling temperature (0.0 = deterministic, 1.5 = creative)
        max_tokens: Maximum number of tokens to generate per response

    Returns:
        A configured LlamaCpp instance, or None if loading fails.

    Raises:
        FileNotFoundError: If the model file doesn't exist at MODEL_PATH
        RuntimeError: If llama.cpp fails to initialize the model
    """
    logger.info(f"Loading model from: {MODEL_PATH}")
    logger.info(f"Context size: {MODEL_CONTEXT_SIZE} tokens")
    logger.info(f"CPU threads: {N_THREADS} | GPU layers: {N_GPU_LAYERS}")

    try:
        llm = LlamaCpp(
            model_path=MODEL_PATH,
            # --- Generation Parameters ---
            temperature=temperature,
            max_tokens=max_tokens,
            # --- Memory & Context ---
            # n_ctx: The context window size in tokens.
            # This is how much text (system prompt + history + current input
            # + response) can fit in memory at once.
            n_ctx=MODEL_CONTEXT_SIZE,
            # --- CPU Configuration ---
            # n_threads: How many CPU cores to use for inference.
            # More threads = faster generation, up to a point.
            n_threads=N_THREADS,
            # --- GPU Configuration ---
            # n_gpu_layers: Number of model layers to offload to GPU.
            # We set 0 because Intel UHD is integrated (no VRAM).
            n_gpu_layers=N_GPU_LAYERS,
            # --- Streaming ---
            # streaming=True allows the model to yield tokens one at a time
            # instead of waiting for the full response.
            streaming=True,
            # --- Reproducibility ---
            # verbose=False suppresses llama.cpp internal logs in production.
            # Set to True for debugging model loading issues.
            verbose=False,
            # stop tokens — TinyLlama Chat uses <|im_end|> as end-of-sequence
            stop=["<|im_end|>", "<|im_start|>", "</s>"],
        )
        logger.info("Model loaded successfully.")
        return llm

    except FileNotFoundError:
        # Re-raise with a helpful message — caught and displayed in app.py
        raise FileNotFoundError(
            f"Model file not found at: {MODEL_PATH}\n"
            f"Please download the GGUF model and place it in models/mistral/\n"
            f"See README.md for download instructions."
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(
            f"Failed to initialize the language model.\n"
            f"Possible causes:\n"
            f"  • Insufficient RAM (need ~4 GB free)\n"
            f"  • Corrupted model file\n"
            f"  • Incompatible llama-cpp-python version\n"
            f"Technical detail: {str(e)}"
        )


def create_llm_with_params(
    temperature: float,
    max_tokens: int,
) -> LlamaCpp:
    """
    Create a new LLM instance with specific parameters.

    Note: This bypasses caching intentionally. Used when the user changes
    generation parameters in the sidebar and wants to apply them immediately.

    The model WEIGHTS are not reloaded — llama.cpp handles parameter changes
    at the inference level.

    Args:
        temperature: New temperature value
        max_tokens: New max_tokens value

    Returns:
        A freshly configured LlamaCpp instance
    """
    return LlamaCpp(
        model_path=MODEL_PATH,
        temperature=temperature,
        max_tokens=max_tokens,
        n_ctx=MODEL_CONTEXT_SIZE,
        n_threads=N_THREADS,
        n_gpu_layers=N_GPU_LAYERS,
        streaming=True,
        verbose=False,
        stop=["<|im_end|>", "<|im_start|>", "</s>"],
    )
