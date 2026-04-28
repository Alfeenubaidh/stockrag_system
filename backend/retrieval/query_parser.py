from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

_get_context_expansions = None


def _load_context_expansions():
    global _get_context_expansions
    if _get_context_expansions is None:
        from retrieval.ranking_signals import get_context_expansions
        _get_context_expansions = get_context_expansions
    return _get_context_expansions


# ---------------------------------------------------------------------------
# Company mapping
# ---------------------------------------------------------------------------

_NAME_TO_TICKER: dict[str, str] = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "intel": "INTC",
    "amd": "AMD",
    "advanced micro devices": "AMD",
    "meta": "META",
}

_KNOWN_TICKERS = frozenset(_NAME_TO_TICKER.values())


# ---------------------------------------------------------------------------
# Intent rules
# ---------------------------------------------------------------------------

_INTENT_RULES = [
    (["non-gaap", "gaap", "adjusted earnings"], "exhibits"),
    (["cybersecurity", "data breach", "cyber"], "cybersecurity"),
    (["acquisition", "antitrust"], "risk factors"),
    (["acquisition", "integration", "risk"], "risk factors"),
    (["gaming", "xbox", "activision"], "gaming"),
    (["outlook", "guidance"], "outlook"),
    (["market risk", "interest rate"], "market risk"),
    (["segment revenue", "business unit"], "segment results"),
    (["production", "delivery"], "md&a"),
    (["vehicle", "delivery"], "md&a"),
    (["product", "service", "business overview"], "business"),
    (["revenue", "earnings", "financial"], "management"),
    (["regulatory", "legal"], "risk factors"),
    (["risk", "competition", "regulatory", "antitrust"], "risk factors"),
]


# ---------------------------------------------------------------------------
# Expansion dictionary
# ---------------------------------------------------------------------------

_EXPANSIONS = {
    "competition": ["competitive landscape", "market competition"],
    "risk": ["risk factors", "material risks"],
    "revenue": ["net sales", "financial performance"],
    "supply chain": ["contract manufacturer", "supplier concentration"],
    "cybersecurity": ["data breach", "cyber incident"],
    "ai": ["artificial intelligence", "machine learning"],
    "training": ["model training", "GPU demand", "data center compute"],
    "inference": ["inference computing", "data center revenue"],
    "data center": ["cloud infrastructure", "GPU demand"],
    "antitrust": ["regulatory approval", "competition law"],
    "gaming": ["Xbox gaming", "game acquisition"],
    "autonomous": ["self-driving", "FSD"],
    "export": ["export controls", "China chip restrictions"],
    "acquisition": ["merger integration", "regulatory merger risk"],
}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

@dataclass
class ParsedQuery:
    original: str
    tickers: list[str]
    section_hint: Optional[str]
    expanded_queries: list[str] = field(default_factory=list)

    inferred_ticker: Optional[str] = None
    confidence: float = 0.0

    @property
    def ticker(self) -> Optional[str]:
        return self.tickers[0] if self.tickers else None


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class QueryParser:

    def parse(self, query: str) -> ParsedQuery:
        q_lower = query.lower()

        explicit_tickers = self._extract_tickers(q_lower)
        section_hint = self._infer_section(q_lower)

        ctx_fn = _load_context_expansions()
        ctx_variants, inferred_ticker, confidence = ctx_fn(q_lower)

        # -------------------------------
        # Tiered ticker handling
        # -------------------------------
        if explicit_tickers:
            active_tickers = explicit_tickers

        elif inferred_ticker:
            if confidence >= 0.85:
                active_tickers = [inferred_ticker]   # hard filter
            else:
                active_tickers = []                 # no hard filter

        else:
            active_tickers = []

        # -------------------------------
        # Expansion (uses inferred ticker)
        # -------------------------------
        variants = self._expand(
            original=query,
            q_lower=q_lower,
            inferred_ticker=inferred_ticker,
            ctx_variants=ctx_variants,
        )

        return ParsedQuery(
            original=query,
            tickers=active_tickers,
            section_hint=section_hint,
            expanded_queries=variants,
            inferred_ticker=inferred_ticker,
            confidence=confidence,
        )

    # ------------------------------------------------------------------

    def _extract_tickers(self, q_lower: str) -> list[str]:
        found = []

        for tok in re.findall(r"\b[A-Z]{2,5}\b", q_lower.upper()):
            if tok in _KNOWN_TICKERS and tok not in found:
                found.append(tok)

        for name, ticker in _NAME_TO_TICKER.items():
            if name in q_lower and ticker not in found:
                found.append(ticker)

        return found

    def _infer_section(self, q_lower: str) -> Optional[str]:
        for keywords, section in _INTENT_RULES:
            if any(kw in q_lower for kw in keywords):
                return section
        return None

    def _expand(
        self,
        original: str,
        q_lower: str,
        inferred_ticker: Optional[str],
        ctx_variants: list[str],
    ) -> list[str]:

        variants = [original]

        # Domain expansions
        extra_terms = []
        for trigger, expansions in _EXPANSIONS.items():
            if trigger in q_lower:
                extra_terms.extend(expansions[:2])

        if extra_terms:
            enriched = original + " " + " ".join(dict.fromkeys(extra_terms))
            if enriched.lower() != original.lower():
                variants.append(enriched)

        # Context expansions
        for v in ctx_variants:
            if v not in variants:
                variants.append(v)

        # Inject ticker for embedding bias (IMPORTANT)
        if inferred_ticker:
            variants.append(f"{inferred_ticker} {original}")

        return variants