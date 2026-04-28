from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional


# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset({
    "a","an","the","and","or","but","in","on","at","to","for","of","with","by","from",
    "is","are","was","were","be","been","have","has","had","do","does","did",
    "will","would","could","should","may","might","can","not","no","nor",
    "its","their","this","that","these","those","what","which","who","how",
    "when","where","why","each","all","any","as","such","than","then",
    "into","through","about","between","during","company","companies",
    "business","including","certain","significant","various","other",
    "also","it","our","we","us","they","them"
})

_MIN_TERM_LEN = 4


# ---------------------------------------------------------------------------
# Boilerplate patterns
# ---------------------------------------------------------------------------

def _section_matches(hint: str, section: str) -> bool:
    if not hint or not section:
        return False
    h = hint.lower().replace(" ", "").replace("_", "").replace("-", "")
    s = section.lower().replace(" ", "").replace("_", "").replace("-", "")
    return h in s or s in h


_BOILERPLATE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"there can be no assurance",
        r"may not achieve",
        r"may not be successful",
        r"material adverse effect",
        r"could result in significant",
        r"we cannot",
        r"no guarantee",
        r"risks and uncertainties",
        r"forward.looking statements",
        r"actual results .* differ",
    ]
]

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# Context expansion rules WITH anchors
# ---------------------------------------------------------------------------

_CONTEXT_EXPANSIONS = [
    (
        ["gaming","acquisition"],
        ["MSFT gaming acquisition"],
        "MSFT",
        0.85,
        ["activision","blizzard","atvi"]
    ),
    (
        ["antitrust","gaming"],
        ["MSFT activision antitrust"],
        "MSFT",
        0.90,
        ["activision","blizzard"]
    ),
    (
        ["digital markets act"],
        ["apple app store dma"],
        "AAPL",
        0.95,
        ["app store","ios"]
    ),
    (
        ["export","chip","china"],
        ["nvidia export control china gpu"],
        "NVDA",
        0.85,
        ["a100","h100"]
    ),
    (
        ["autonomous"],
        ["TSLA autonomous vehicle FSD regulatory", "Tesla self-driving software regulation"],
        "TSLA",
        0.85,
        [],
    ),
    (
        ["production","delivery"],
        ["TSLA vehicle production delivery units", "Tesla annual delivery production numbers"],
        "TSLA",
        0.85,
        [],
    ),
    (
        ["vehicle","delivery"],
        ["TSLA vehicle production delivery units", "Tesla annual delivery production numbers"],
        "TSLA",
        0.85,
        [],
    ),
    (
        ["contract manufacturer"],
        ["AAPL contract manufacturer Foxconn device assembly", "Apple single supplier concentration risk assembly"],
        "AAPL",
        0.85,
        [],
    ),
    (
        ["manufacturer","assembly"],
        ["AAPL contract manufacturer Foxconn device assembly", "Apple single supplier concentration risk assembly"],
        "AAPL",
        0.85,
        [],
    ),
]


# ---------------------------------------------------------------------------
# Context expansion with FIXED conflict handling
# ---------------------------------------------------------------------------

def get_context_expansions(query_lower: str):
    expansions = []
    ticker_signals = []

    for required, extra, ticker, conf, anchors in _CONTEXT_EXPANSIONS:
        if all(r in query_lower for r in required):
            expansions.extend(extra)
            expansions.extend(anchors)

            if ticker:
                ticker_signals.append((ticker, conf))

    if not ticker_signals:
        return expansions, None, 0.0

    # aggregate best confidence per ticker
    ticker_scores = {}
    for t, c in ticker_signals:
        ticker_scores[t] = max(ticker_scores.get(t, 0), c)

    sorted_tickers = sorted(ticker_scores.items(), key=lambda x: x[1], reverse=True)

    top_ticker, top_conf = sorted_tickers[0]

    if len(sorted_tickers) > 1:
        second_conf = sorted_tickers[1][1]
        if top_conf - second_conf < 0.15:
            return expansions, None, top_conf  # keep signal, no hard filter

    return expansions, top_ticker, top_conf


# ---------------------------------------------------------------------------
# Ranking scorer
# ---------------------------------------------------------------------------

class RankingSignalScorer:

    KEYWORD_WEIGHT = 2.0
    CO_OCCUR_WEIGHT = 1.5
    BOILERPLATE_WEIGHT = 1.5

    def score(
        self,
        query: str,
        chunk_text: str,
        chunk_ticker: Optional[str] = None,
        inferred_ticker: Optional[str] = None,
        confidence: float = 0.0,
        section_hint: Optional[str] = None,
        chunk_section: Optional[str] = None,
    ):
        query_terms = _extract_terms(query)
        if not query_terms:
            return 0.0, 0.0

        text_lower = chunk_text.lower()

        bonus = self._keyword_bonus(query_terms, text_lower)
        penalty = self._boilerplate_penalty(chunk_text)

        # ENTITY-AWARE SCORING
        if inferred_ticker and chunk_ticker:
            if chunk_ticker == inferred_ticker:
                if confidence >= 0.85:
                    bonus += 0.5
                elif confidence >= 0.7:
                    bonus += 0.3
            else:
                bonus -= 0.2

        if section_hint and chunk_section and _section_matches(section_hint, chunk_section):
            bonus += 1.0

        return bonus, penalty


    def _keyword_bonus(self, query_terms, text_lower):
        matched = [t for t in query_terms if t in text_lower]
        if not matched:
            return 0.0

        overlap = len(matched) / len(query_terms)
        overlap_bonus = self.KEYWORD_WEIGHT * overlap

        co_bonus = 0.0
        if len(matched) >= 2:
            co_bonus = self.CO_OCCUR_WEIGHT * (len(matched)-1) / len(query_terms)

        return overlap_bonus + co_bonus


    def _boilerplate_penalty(self, text):
        sentences = _SENTENCE_RE.split(text.strip())
        if not sentences:
            return 0.0

        count = sum(
            1 for s in sentences
            if any(p.search(s) for p in _BOILERPLATE_PATTERNS)
        )

        fraction = count / len(sentences)

        if fraction > 0.3:
            return self.BOILERPLATE_WEIGHT * fraction + 0.2

        return self.BOILERPLATE_WEIGHT * fraction


# ---------------------------------------------------------------------------
# Term extraction
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _extract_terms(query: str):
    tokens = re.findall(r"[a-z]+", query.lower())
    terms = [t for t in tokens if t not in _STOPWORDS and len(t) >= _MIN_TERM_LEN]

    for i in range(len(terms)-1):
        terms.append(f"{terms[i]} {terms[i+1]}")

    return list(dict.fromkeys(terms))