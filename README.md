# 🤖 Local Mistral Chatbot — v0.2

A fully local conversational AI chatbot built with **Python**, **LangChain**, **Mistral 7B (GGUF)**, and **Streamlit**.

**No cloud API. No internet required. Your conversations never leave your machine.**

---

## ✨ Features

- 💬 **Multi-turn conversation** — full context memory across turns
- ⚡ **Streaming responses** — see tokens generated in real time
- 🎛️ **Configurable generation** — temperature and max tokens via sliders
- 📋 **Custom system prompt** — change the assistant's persona at runtime
- 🔄 **New conversation** — clear history with one click
- 🏗️ **RAG-ready architecture** — v0.3 can add retrieval without a rewrite
- 🎨 **Custom dark UI** — glassmorphism design with animated message bubbles

---

## 🖥️ System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 6 GB free | 8 GB+ |
| CPU | x86_64, 4+ cores | 8+ cores |
| GPU | Not required | Optional (NVIDIA for faster inference) |
| OS | Windows 10+, Linux, macOS | Any |
| Python | 3.10+ | 3.10–3.12 |
| Disk | 4 GB free | 8 GB+ |

### Model RAM Requirements (Mistral 7B)

| Quantization | File Size | RAM at Runtime | Recommended For |
|---|---|---|---|
| Q2_K | ~2.7 GB | ~3.5 GB | ≤8 GB RAM |
| **Q3_K_S** | **~3.1 GB** | **~4.0 GB** | **≤8 GB RAM ✅ (this build)** |
| Q3_K_M | ~3.5 GB | ~4.5 GB | 8–12 GB RAM |
| Q4_K_M | ~4.4 GB | ~5.5 GB | 12 GB+ RAM |

---

## 🚀 Installation

### 1. Clone / navigate to the project

```bash
cd local-mistral-chatbot
```

### 2. Create the virtual environment

```bash
uv venv --python 3.10
```

### 3. Install llama-cpp-python (CPU pre-built wheel — no compiler needed)

```bash
uv pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

> **Note**: If you have an NVIDIA GPU and CUDA installed, use the CUDA wheel instead for much faster inference. See [llama-cpp-python releases](https://github.com/abetlen/llama-cpp-python/releases).

### 4. Install remaining dependencies

```bash
uv pip install "langchain>=0.3" "langchain-core>=0.3" "langchain-community>=0.3" "streamlit>=1.40" "python-dotenv>=1.0"
```

### 5. Configure environment

```bash
copy .env.example .env
```

Edit `.env` if needed (the defaults work for most setups).

---

## 📥 Model Setup

Download the **Mistral 7B Instruct v0.2 Q3_K_S** GGUF from Hugging Face:

**URL**: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF

**File to download**: `mistral-7b-instruct-v0.2.Q3_K_S.gguf` (~3.1 GB)

Place it here:

```
local-mistral-chatbot/
└── models/
    └── mistral/
        └── mistral-7b-instruct-v0.2.Q3_K_S.gguf   ← here
```

> The model file is gitignored and will never be accidentally committed.

---

## ▶️ Running the Application

```bash
uv run streamlit run app.py
```

Then open your browser at: **http://localhost:8501**

---

## ⚙️ Configuration

All settings are in `.env`. Key variables:

| Variable | Default | Description |
|---|---|---|
| `MODEL_FILENAME` | `mistral-7b-instruct-v0.2.Q3_K_S.gguf` | Model file name |
| `MODEL_CONTEXT_SIZE` | `4096` | Context window in tokens |
| `DEFAULT_TEMPERATURE` | `0.7` | Generation randomness |
| `DEFAULT_MAX_TOKENS` | `512` | Max response length |
| `N_THREADS` | `8` | CPU threads for inference |
| `N_GPU_LAYERS` | `0` | GPU layers (0 = CPU only) |
| `MAX_HISTORY_TURNS` | `10` | Conversation turns to keep |

Runtime controls (temperature, max tokens, system prompt) can also be changed in the Streamlit sidebar without restarting.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│         Streamlit UI (app.py)       │  User-facing layer only
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│     LangChain LCEL Chain            │
│   prompt | llm | parser             │  Orchestration (chains.py)
└──────┬──────────┬────────┬──────────┘
       │          │        │
  prompts.py  llm.py   parsers.py
  (Template) (LlamaCpp) (StrOutput)
       │
  memory.py
  (ChatMessageHistory)
       │
┌──────▼──────────────────────────────┐
│   mistral-7b-instruct-v0.2.Q3_K_S   │
│   GGUF · llama.cpp · CPU inference  │
└─────────────────────────────────────┘
```

### File Responsibilities

| File | Responsibility |
|---|---|
| `app.py` | Streamlit UI, session state, rendering |
| `src/config.py` | All settings and constants |
| `src/llm.py` | Model loading, caching, LlamaCpp config |
| `src/prompts.py` | ChatPromptTemplate, Mistral Instruct format |
| `src/memory.py` | ChatMessageHistory, history trimming |
| `src/chains.py` | LCEL pipeline, streaming, invoke |
| `src/parsers.py` | StrOutputParser, output cleaning |
| `src/utils.py` | Validation, helpers, system info |

---

## 🔬 Key Concepts

### Why GGUF?
GGUF is a compact binary format for quantized LLM weights. It allows running large models on consumer hardware by reducing precision (e.g. from 16-bit to 3-bit per weight).

### Why llama.cpp?
A highly optimized C++ inference engine for running GGUF models on CPU (with optional GPU offloading). Uses SIMD instructions (AVX2/AVX512) to maximize throughput on modern CPUs.

### Why LangChain?
LangChain provides composable abstractions (prompt templates, chains, memory, parsers) that make it easy to wire together LLM pipelines and extend them later (e.g. adding RAG).

### Why LCEL?
LangChain Expression Language (`prompt | llm | parser`) makes chains declarative and streaming-native. Every component is a `Runnable` — swappable without touching the rest.

### Why Streamlit?
Pure Python web UI with no JavaScript required. `st.cache_resource` ensures the 3 GB model is loaded once and shared across all interactions.

---

## 🔧 Troubleshooting

### Model file not found
```
❌ Model file not found at: models/mistral/mistral-7b-instruct-v0.2.Q3_K_S.gguf
```
**Fix**: Download the GGUF file and place it in `models/mistral/`.

### Generation is very slow
**Expected**: CPU-only inference on Mistral 7B yields ~3–8 tokens/second. This is normal.
- Try a lower quantization (Q2_K) for slightly faster speed
- Close other RAM-heavy applications to free memory

### `llama-cpp-python` import error
**Fix**: Reinstall using the pre-built wheel:
```bash
uv pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --force-reinstall
```

### Out of memory crash
**Fix**: Lower `MODEL_CONTEXT_SIZE` in `.env` to `2048`, or switch to `Q2_K`.

### Responses cut off mid-sentence
**Fix**: Increase `DEFAULT_MAX_TOKENS` in `.env` or use the sidebar slider.

---

## 🔮 v0.3 — RAG Roadmap

v0.3 will add Retrieval-Augmented Generation. The architecture is already designed for it.

**New files** (v0.3 only):
- `src/embeddings.py` — local embedding model
- `src/vectorstore.py` — ChromaDB / FAISS vector store
- `src/retriever.py` — document retrieval

**Modified files** (minimal changes):
- `src/chains.py` — inject retrieved context into pipeline
- `src/prompts.py` — add `{context}` slot to template

Everything else stays the same.

```
Documents → Loader → Splitter → Embedder → VectorStore
                                                  ↓
User Query → Retriever → Context → Prompt → Mistral → Response
```

---

## 📄 License

MIT License — free to use, modify, and distribute.
