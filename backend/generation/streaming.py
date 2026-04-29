from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from market_data import get_snapshot
from .generator import (
    _MAX_TOKENS,
    _MODEL,
    _TEMPERATURE,
    _get_client,
)

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_stream_prompts() -> tuple[str, str]:
    system = (_PROMPTS_DIR / "stream_system_prompt.txt").read_text(encoding="utf-8").strip()
    answer = (_PROMPTS_DIR / "stream_answer_prompt.txt").read_text(encoding="utf-8").strip()
    return system, answer


def _build_context(chunks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for c in chunks:
        ticker = c.get("ticker", "")
        doc_type = c.get("doc_type", "10-K")
        section = c.get("section", "")
        text = c.get("text", "").strip()
        if not text:
            continue
        header = f"[{ticker} {doc_type} · {section}]" if section else f"[{ticker} {doc_type}]"
        parts.append(f"{header}\n{text}")
    return "\n\n".join(parts)


def stream_answer(
    query: str,
    retrieved_chunks: list[dict[str, Any]],
) -> Iterator[str]:
    """
    Stream LLM answer tokens from Groq as they are generated.

    Yields raw token strings as they arrive. The caller is responsible for SSE framing.
    """
    if not retrieved_chunks:
        logger.warning("stream_answer: empty retrieved_chunks, yielding empty response")
        yield "No information found in retrieved filings."
        return

    context = _build_context(retrieved_chunks)
    if not context:
        logger.warning("stream_answer: context build returned empty string")
        yield "No information found in retrieved filings."
        return

    ticker = retrieved_chunks[0].get("ticker") if retrieved_chunks else None
    market_block = get_snapshot(ticker) if ticker else None
    if market_block:
        context = market_block + "\n\n" + context

    system_prompt, answer_template = _load_stream_prompts()
    user_message = answer_template.format(query=query, context=context)

    try:
        stream = _get_client().chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=_MAX_TOKENS,
            temperature=_TEMPERATURE,
            stream=True,
        )
        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                yield token
    except Exception as exc:
        logger.error("stream_answer: Groq API error: %s", exc)
        yield "\n\n[Generation error: Groq API unavailable]"
