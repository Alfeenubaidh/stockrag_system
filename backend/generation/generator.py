from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import os

from groq import Groq

from market_data import get_snapshot
from .citation_assembler import CitationValidationError, assemble_and_validate
from .grounding_check import check_grounding
from .context_builder import build_context
from generation.citation_postprocessor import CitationPostProcessor, RetrievedChunk
from observability.pipeline_observer import observer

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_SYSTEM_PROMPT_PATH = _PROMPTS_DIR / "system_prompt.txt"
_ANSWER_PROMPT_PATH = _PROMPTS_DIR / "answer_prompt.txt"

_postprocessor = CitationPostProcessor(min_match_threshold=0.15, max_citations_per_sentence=2)

_EMPTY_RESPONSE: dict[str, Any] = {
    "answers": {},
    "comparison": "Not found in filings",
}

_DISCLAIMER = "This is not financial advice. For informational purposes only."

_MODEL = "llama-3.1-8b-instant"
_MAX_TOKENS = 512
_TEMPERATURE = 0  # deterministic
_MAX_CHUNK_CHARS = 1000  # ~250 tokens; keeps all chunks within 4096 ctx window

_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Extraction pre-pass: bridges paraphrase gap for regulatory/availability queries.
# Triggers when query contains any of these terms, runs on chunks scoring above threshold.
_EXTRACTION_KEYWORDS: frozenset[str] = frozenset({
    "regulatory", "regulation", "approval", "approving",
    "availability", "autonomous", "restrict", "delay",
})
_EXTRACTION_SCORE_THRESHOLD = 0.7
_EXTRACTION_SYSTEM = "You are a precise text analyst. Answer concisely."
# ~45 tokens — well under the 100-token budget
_EXTRACTION_PROMPT = (
    "Does this text mention delays in regulatory approval or restrictions on "
    "product availability? Reply yes or no. If yes, quote the exact phrase.\n\nText:\n{text}"
)

# Cached prompt strings (loaded once per process)
_system_prompt: str | None = None
_answer_prompt_template: str | None = None


def _load_prompts() -> tuple[str, str]:
    global _system_prompt, _answer_prompt_template
    if _system_prompt is None:
        _system_prompt = _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    if _answer_prompt_template is None:
        _answer_prompt_template = _ANSWER_PROMPT_PATH.read_text(encoding="utf-8")
    return _system_prompt, _answer_prompt_template


def _format_structured_context(context: dict[str, list[dict[str, str]]]) -> str:
    """
    Render the grouped context dict into a human-readable string for the prompt.

    Format per chunk:
        [AAPL | Risk Factors]
        <text>
    """
    lines: list[str] = []
    for ticker, chunks in context.items():
        for chunk in chunks:
            section = chunk.get("section", "Unknown Section")
            text = chunk.get("text", "").strip()
            if len(text) > _MAX_CHUNK_CHARS:
                text = text[:_MAX_CHUNK_CHARS] + " [truncated]"
                logger.debug("Chunk truncated: ticker=%s section=%s", ticker, section)
            lines.append(f"[{ticker} | {section}]")
            lines.append(text)
            lines.append("")  # blank line separator
    return "\n".join(lines).strip()


def _parse_llm_response(raw_text: str) -> dict[str, Any]:
    """
    Extract and parse JSON from LLM response.
    Handles bare JSON, JSON in markdown fences, and JSON wrapped in prose.

    Raises:
        ValueError: if JSON cannot be parsed.
    """
    # Try markdown fences first
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw_text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try bare JSON
    try:
        return json.loads(raw_text.strip())
    except json.JSONDecodeError:
        pass

    # Fallback: extract substring from first '{' to last '}'
    brace_match = re.search(r"\{[\s\S]*\}", raw_text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

        # SEC filings often contain parenthetical abbreviations like ("CGC") which
        # the LLM reproduces verbatim — inner quotes break JSON. Escape them and retry.
        sanitized = re.sub(r'\("([^"]+)"\)', r'(\\"\1\\")', brace_match.group(0))
        try:
            return json.loads(sanitized)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"LLM response is not valid JSON.\nRaw response:\n{raw_text}")


def _validate_schema(parsed: dict) -> None:
    """
    Enforce top-level schema: must have 'answers' (dict) and 'comparison' (str).

    Raises:
        ValueError: on schema mismatch.
    """
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected dict from LLM, got {type(parsed)}")
    if "answers" not in parsed:
        raise ValueError("Missing 'answers' key in LLM output")
    if not isinstance(parsed["answers"], dict):
        raise ValueError(f"'answers' must be a dict, got {type(parsed['answers'])}")
    if "comparison" in parsed and not isinstance(parsed["comparison"], str):
        raise ValueError(f"'comparison' must be a str, got {type(parsed['comparison'])}")


def _call_llm(
    system_prompt: str,
    user_message: str,
) -> str:
    """Call Groq API. Returns the raw response text."""
    response = _client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        max_tokens=_MAX_TOKENS,
        temperature=_TEMPERATURE,
    )
    return response.choices[0].message.content or ""


def _dedup_citations(text: str) -> str:
    # Remove consecutive duplicate citation tags e.g. [AAPL, Risk Factors] [AAPL, Risk Factors]
    return re.sub(r'(\[[A-Z]+,\s*[^\]]+\])(\s*\1)+', r'\1', text)


def _extraction_triggered(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in _EXTRACTION_KEYWORDS)


def _extract_relevant_phrase(chunk_text: str) -> str | None:
    """
    Ask the LLM whether the chunk mentions regulatory delays or availability
    restrictions, and return the quoted phrase if it does.
    Returns None on 'no' answer or any failure — never raises.
    """
    prompt = _EXTRACTION_PROMPT.format(text=chunk_text[:1500])
    try:
        response = _call_llm(_EXTRACTION_SYSTEM, prompt)
    except Exception as exc:
        logger.debug("Extraction pre-pass failed (skipping): %s", exc)
        return None
    if not response.lower().startswith("yes"):
        return None
    quoted = re.search(r'"([^"]{10,})"', response)
    if quoted:
        return quoted.group(1).strip()
    # Fallback: strip the leading yes/no token and take the remainder
    after = re.sub(r'^yes[.,]?\s*', '', response, flags=re.IGNORECASE).strip()
    return after[:300] if after else None


def generate_answer(
    query: str,
    retrieved_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Generate a grounded, citation-enforced answer from retrieved chunks.

    Args:
        query: The user's question.
        retrieved_chunks: Raw retrieval results from QueryInterface.

    Returns:
        {
            "answers": {"AAPL": "...", "MSFT": "..."},
            "comparison": "..."
        }
        On insufficient context or unrecoverable errors, returns _EMPTY_RESPONSE.

    Raises:
        Exception: On Groq API errors.
    """
    if not retrieved_chunks:
        logger.warning("generate_answer: empty retrieved_chunks, returning empty response.")
        return _EMPTY_RESPONSE

    # Step 1: Build and validate context
    try:
        context = build_context(retrieved_chunks)
    except Exception as exc:
        logger.error("context_builder failed: %s", exc, exc_info=True)
        return _EMPTY_RESPONSE

    if not context:
        logger.warning("generate_answer: context_builder returned empty context.")
        return _EMPTY_RESPONSE

    # Step 1b: Extraction pre-pass — bridges paraphrase gap for regulatory/availability queries.
    # Runs a cheap yes/no extraction on high-scoring chunks; prepends any matched phrases to
    # the generation context so the main LLM call has explicit bridging evidence.
    extraction_notes: list[str] = []
    if _extraction_triggered(query):
        high_score_chunks = [c for c in retrieved_chunks if c.get("score", 0.0) > _EXTRACTION_SCORE_THRESHOLD]
        if high_score_chunks:
            for chunk in high_score_chunks:
                phrase = _extract_relevant_phrase(chunk.get("text", ""))
                if phrase:
                    ticker = chunk.get("ticker", "?")
                    extraction_notes.append(f'[{ticker}] Relevant phrase: "{phrase}"')
            if extraction_notes:
                logger.info("Extraction pre-pass: %d phrase(s) found", len(extraction_notes))
            else:
                logger.debug("Extraction pre-pass: no matching phrases in %d chunk(s)", len(high_score_chunks))

    # Step 2: Load prompts
    try:
        system_prompt, answer_prompt_template = _load_prompts()
    except OSError as exc:
        logger.error("Failed to load prompts: %s", exc, exc_info=True)
        raise  # Prompt loading failure is a deployment error, not a runtime fallback

    structured_context_str = _format_structured_context(context)

    ticker = next(iter(context), None)
    market_block = get_snapshot(ticker) if ticker else None
    if market_block:
        structured_context_str = market_block + "\n\n" + structured_context_str

    if extraction_notes:
        extraction_header = (
            "Key phrases extracted from the filings (use these to ground your answer):\n"
            + "\n".join(extraction_notes)
            + "\n\n"
        )
        structured_context_str = extraction_header + structured_context_str

    user_message = answer_prompt_template.format(
        query=query,
        structured_context=structured_context_str,
    )

    # Step 3: Call LLM
    try:
        raw_text = _call_llm(system_prompt, user_message)
    except Exception as exc:
        reason = f"Groq API error: {exc}"
        logger.error(reason)
        return {**_EMPTY_RESPONSE, "error": reason}

    if not raw_text.strip():
        logger.warning("generate_answer: Groq returned empty response.")
        return _EMPTY_RESPONSE

    # Step 4: Parse and validate JSON schema
    try:
        parsed = _parse_llm_response(raw_text)
    except ValueError as exc:
        reason = f"JSON parse failed: {exc}"
        logger.error(reason)
        return {**_EMPTY_RESPONSE, "error": reason}

    try:
        _validate_schema(parsed)
    except ValueError as exc:
        reason = f"Schema validation failed: {exc}"
        logger.error(reason)
        return {**_EMPTY_RESPONSE, "error": reason}

    # Step 5: Citation post-processing — repair/inject citations before validation
    # Build per-ticker metadata for citation format: [TICKER DOC_TYPE · FYYYY · Section]
    ticker_metadata: dict[str, dict] = {}
    for chunk in retrieved_chunks:
        t = chunk.get("ticker", "").upper()
        if t and t not in ticker_metadata:
            ticker_metadata[t] = {
                "doc_type": chunk.get("doc_type", ""),
                "filing_date": chunk.get("filing_date", ""),
            }

    chunks = [
        RetrievedChunk(
            ticker=r.get("ticker", ""),
            section=r.get("section", ""),
            text=r.get("text", ""),
            score=r.get("score", 0.0),
            doc_type=r.get("doc_type", ""),
            filing_date=r.get("filing_date", ""),
        )
        for r in retrieved_chunks
    ]
    total_added = 0
    total_removed = 0
    corrected_answers: dict[str, Any] = {}
    for ticker, answer_text in parsed.get("answers", {}).items():
        if not isinstance(answer_text, str):
            corrected_answers[ticker] = answer_text
            continue
        ticker_chunks = [c for c in chunks if c.ticker == ticker] or chunks
        result = _postprocessor.process(answer_text, ticker_chunks)
        corrected_answers[ticker] = _dedup_citations(result.corrected)
        total_added += result.citations_added
        total_removed += result.citations_removed
    repaired = {**parsed, "answers": corrected_answers}

    # Step 6: Citation enforcement on repaired output
    # Log before validation so failures are always traceable.
    observer.log_generation(
        prompt=user_message,
        raw_output=raw_text,
        corrected_output=str(repaired),
        citations_added=total_added,
        citations_removed=total_removed,
        answer_tickers=list(repaired.get("answers", {}).keys()),
    )

    try:
        validated = assemble_and_validate(repaired, ticker_metadata=ticker_metadata)
    except CitationValidationError as exc:
        reason = f"Citation validation failed: {exc}"
        logger.error(reason)
        return {**_EMPTY_RESPONSE, "error": reason}

    # Step 7: Per-ticker completeness — every ticker present in context must appear in answers
    for ticker in context:
        if ticker not in validated["answers"]:
            logger.warning(
                "generate_answer: ticker %s absent from LLM output; inserting placeholder.",
                ticker,
            )
            validated["answers"][ticker] = "No information found in retrieved filings."

    # Step 8: Mandatory disclaimer — every generated response must carry this.
    validated["disclaimer"] = _DISCLAIMER

    # Step 9: Grounding check — verify each claim against retrieved chunks.
    validated = check_grounding(validated, retrieved_chunks)

    return validated