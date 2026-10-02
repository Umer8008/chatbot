"""
prompts.py — Prompt Templates


This module defines how we structure the conversation before sending it to
the model. This is one of the most important parts of any LangChain application.

WHY PROMPT TEMPLATES?
  Raw strings are fragile — easy to introduce formatting bugs, hard to
  maintain, and impossible to validate. LangChain's prompt templates:
  1. Separate the "structure" from the "content"
  2. Handle variable substitution safely
  3. Make the prompt testable and inspectable
  4. Allow swapping templates without touching chain logic

WHY TINYLLAMA CHATML FORMAT?
  TinyLlama Chat v1.0 was fine-tuned on the ChatML format:
    <|im_start|>system
    {system_prompt}<|im_end|>
    <|im_start|>user
    {user_message}<|im_end|>
    <|im_start|>assistant

  Using any other format causes the model to produce irrelevant or
  incoherent responses because it doesn't recognise the turn boundaries.

WHY MessagesPlaceholder?
  MessagesPlaceholder is a slot in the template that accepts a LIST of
  messages at runtime. This is how we inject conversation history —
  we pass the accumulated [HumanMessage, AIMessage, HumanMessage, ...] list
  into this placeholder and the template expands it correctly.

  Without this, we couldn't support multi-turn conversations.
"""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import PromptTemplate

from src.config import DEFAULT_SYSTEM_PROMPT


def build_chatml_prompt(
    history: list,
    user_input: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """
    Build a raw ChatML-format prompt string for TinyLlama.

    TinyLlama Chat v1.0 expects this exact format:
        <|im_start|>system
        {system}<|im_end|>
        <|im_start|>user
        {msg}<|im_end|>
        <|im_start|>assistant
        {reply}<|im_end|>
        ...
        <|im_start|>user
        {current_input}<|im_end|>
        <|im_start|>assistant

    The trailing <|im_start|>assistant (without <|im_end|>) signals the model
    to begin generating the assistant's reply.

    Args:
        history:       List of past HumanMessage / AIMessage objects
        user_input:    The current user's message
        system_prompt: The system-level instruction

    Returns:
        A fully formatted prompt string ready to be sent to LlamaCpp.
    """
    parts = []

    # System turn
    parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")

    # Conversation history
    for msg in history:
        if isinstance(msg, HumanMessage):
            parts.append(f"<|im_start|>user\n{msg.content}<|im_end|>")
        elif isinstance(msg, AIMessage):
            parts.append(f"<|im_start|>assistant\n{msg.content}<|im_end|>")

    # Current user turn + open assistant turn to trigger generation
    parts.append(f"<|im_start|>user\n{user_input}<|im_end|>")
    parts.append("<|im_start|>assistant")

    return "\n".join(parts)


def get_chat_prompt(system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> PromptTemplate:
    """
    Returns a LangChain PromptTemplate that wraps build_chatml_prompt.
    Used by get_chain() so the LCEL pipe (prompt | llm | parser) still works.

    NOTE: history and input are injected via build_chatml_prompt; this
    template accepts a single pre-formatted 'prompt' variable.
    """
    return PromptTemplate.from_template("{prompt}")


def get_system_prompt_preview(system_prompt: str, max_chars: int = 100) -> str:
    """
    Return a truncated preview of the system prompt for UI display.

    Args:
        system_prompt: The full system prompt string
        max_chars: Maximum characters to show

    Returns:
        Truncated string with ellipsis if needed
    """
    if len(system_prompt) <= max_chars:
        return system_prompt
    return system_prompt[:max_chars].rstrip() + "..."
