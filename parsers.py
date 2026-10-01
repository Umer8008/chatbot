"""
parsers.py — Output Parsers

This module demonstrates LangChain's output parsing layer.

WHY OUTPUT PARSERS?
  When the LLM generates a response, it returns a raw object —
  a LangChain AIMessage or a string, depending on the LLM type.
  Output parsers transform this raw output into whatever format
  your application needs.

  Think of it like this:
    Raw model output   → "The answer is 42\n\n"  (AIMessage or str with noise)
    StrOutputParser    → "The answer is 42"       (clean Python string)
    JsonOutputParser   → {"answer": 42}            (parsed Python dict)
    PydanticParser     → MyModel(answer=42)         (typed Pydantic object)

  For a conversational chatbot, StrOutputParser is exactly what we need —
  it extracts the text content from the model's response and strips
  any unnecessary whitespace.

  For v0.3 RAG, we might want a structured parser that returns both
  the answer AND the source documents used.

DIFFERENCE BETWEEN RAW AND PARSED OUTPUT:
  Without parser: chain returns AIMessage(content="Hello!", ...)
  With StrOutputParser: chain returns "Hello!"

  The parser sits at the END of the LCEL chain:
    prompt | llm | parser
  So the final output of chain.invoke() or chain.stream() is always
  a clean string, ready for display.
"""

from langchain_core.output_parsers import StrOutputParser


def get_string_parser() -> StrOutputParser:
    """
    Return a StrOutputParser — the standard parser for conversational chatbots.

    StrOutputParser does three things:
    1. Extracts the text content from AIMessage objects
    2. Passes through plain strings unchanged
    3. Strips leading/trailing whitespace

    This is the correct parser for our use case because:
    - We want clean strings to display in the Streamlit UI
    - We don't need JSON or structured data for basic chat
    - It's fully compatible with streaming (works token by token)

    Returns:
        StrOutputParser instance ready to use in an LCEL chain
    """
    return StrOutputParser()


# ─────────────────────────────────────────────────────────────────────────────
# EDUCATIONAL SECTION — Structured Output (not used in normal chat)
# ─────────────────────────────────────────────────────────────────────────────
# In v0.3 or specialized features, you might want structured output.
# Example: extracting a sentiment label alongside the response.
#
# from langchain_core.output_parsers import JsonOutputParser
# from pydantic import BaseModel, Field
#
# class StructuredResponse(BaseModel):
#     answer: str = Field(description="The assistant's response")
#     confidence: str = Field(description="high, medium, or low")
#
# parser = JsonOutputParser(pydantic_object=StructuredResponse)
#
# You would then change your prompt to instruct the model to output JSON,
# and your chain would become:
#   chain = prompt | llm | parser
#
# The output would be: StructuredResponse(answer="...", confidence="high")
# ─────────────────────────────────────────────────────────────────────────────
