"""
memory.py — Conversation History Management

This module manages conversation history for multi-turn chat.

WHY CONVERSATION HISTORY?
  LLMs are stateless by default — each call is independent.
  To support multi-turn dialogue ("What is my name?" after "My name is Ali"),
  we must manually maintain and inject past messages into every new prompt.

  The flow is:
    Turn 1: [System] + [Human: "My name is Ali."] → LLM → [AI: "Nice to meet you, Ali."]
    Turn 2: [System] + [Human: "My name is Ali."] + [AI: "Nice to meet you, Ali."]
                     + [Human: "What is my name?"] → LLM → [AI: "Your name is Ali."]

  Each turn we append to the history and pass THE ENTIRE accumulated
  history to the model. The model then has full context.

WHY ChatMessageHistory?
  LangChain's ChatMessageHistory is a simple, structured container for
  HumanMessage and AIMessage objects. It's the standard way to store
  conversation turns in LangChain v0.3+.

  We store it in Streamlit's session_state so it persists across
  UI interactions within a single browser session.

WHY SESSION STATE (not a database)?
  v0.2 is a single-user local app. Session state is perfect here —
  it's in-memory, fast, and automatically cleared when the user
  starts a new session or clicks "New Chat".

  In v0.3+, if we wanted persistent history across sessions, we would
  swap ChatMessageHistory for SQLChatMessageHistory (LangChain's SQL adapter).
  The rest of the code would not change — just the storage backend.

HISTORY TRIMMING:
  Long conversations can exceed the model's context window (4096 tokens).
  We keep only the last N turns to prevent context overflow.
"""

from typing import List

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.config import MAX_HISTORY_TURNS


def create_message_history() -> ChatMessageHistory:
    """
    Create a new, empty ChatMessageHistory instance.

    This is called when the app starts or when the user clicks "New Chat".

    Returns:
        An empty ChatMessageHistory object.
    """
    return ChatMessageHistory()


def add_user_message(history: ChatMessageHistory, content: str) -> None:
    """
    Append the user's message to conversation history.

    Args:
        history: The current ChatMessageHistory instance
        content: The user's message text
    """
    history.add_user_message(content)


def add_ai_message(history: ChatMessageHistory, content: str) -> None:
    """
    Append the assistant's response to conversation history.

    Args:
        history: The current ChatMessageHistory instance
        content: The assistant's response text
    """
    history.add_ai_message(content)


def get_trimmed_messages(history: ChatMessageHistory) -> List[BaseMessage]:
    """
    Return the conversation history, trimmed to the last N turns.

    Why trim?
      Each token in the history consumes context window space.
      Our model has a 4096-token context window. A long conversation
      can overflow this, causing errors or the model to "forget" the
      beginning of the window.

      By keeping only the last MAX_HISTORY_TURNS (default: 10) turns,
      we ensure the prompt always fits within the context window.

    Technical note:
      Each "turn" = 1 HumanMessage + 1 AIMessage = 2 messages.
      So MAX_HISTORY_TURNS=10 → keep last 20 messages.

    Args:
        history: The current ChatMessageHistory instance

    Returns:
        A list of BaseMessage objects (HumanMessage + AIMessage pairs),
        trimmed to the most recent MAX_HISTORY_TURNS turns.
    """
    messages = history.messages
    # Each turn is 2 messages (human + ai), so multiply
    max_messages = MAX_HISTORY_TURNS * 2
    if len(messages) > max_messages:
        # Keep the most recent messages
        return messages[-max_messages:]
    return messages


def clear_history(history: ChatMessageHistory) -> None:
    """
    Clear all messages from the conversation history.

    Called when the user clicks "New Chat" in the Streamlit sidebar.

    Args:
        history: The ChatMessageHistory instance to clear
    """
    history.clear()


def get_turn_count(history: ChatMessageHistory) -> int:
    """
    Return the number of complete conversation turns (user+assistant pairs).

    Args:
        history: The current ChatMessageHistory instance

    Returns:
        Number of complete turns (integer)
    """
    # Divide total messages by 2 (each turn = human + ai message)
    return len(history.messages) // 2


def format_history_for_display(history: ChatMessageHistory) -> List[dict]:
    """
    Convert ChatMessageHistory into a simple list of dicts for Streamlit display.

    Streamlit's chat UI works best with {"role": ..., "content": ...} dicts.
    This function bridges LangChain's message objects with Streamlit's format.

    Args:
        history: The current ChatMessageHistory instance

    Returns:
        List of {"role": "user"/"assistant", "content": str} dicts
    """
    result = []
    for msg in history.messages:
        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            result.append({"role": "assistant", "content": msg.content})
    return result
