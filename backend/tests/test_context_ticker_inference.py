"""
tests/test_context_ticker_inference.py — Unit + integration tests for
context-aware ticker inference.

Tests two layers:
  1. get_context_expansions() — verifies (expansions, ticker, confidence) output.
  2. QueryParser.parse()      — verifies that the threshold gate (>= 0.8) is
                                applied correctly and explicit tickers always win.

Run with:
    pytest tests/test_context_ticker_inference.py -v
"""
from __future__ import annotations

import pytest

from retrieval.ranking_signals import get_context_expansions
from retrieval.query_parser import QueryParser

THRESHOLD = 0.8


# ---------------------------------------------------------------------------
# get_context_expansions — unit tests
# ---------------------------------------------------------------------------

class TestGetContextExpansions:

    def test_gaming_acquisition_antitrust_infers_msft_high_confidence(self):
        # Both gaming+acquisition (0.85) and antitrust+gaming (0.90) rules fire.
        # max confidence = 0.90, all signals agree on MSFT.
        expansions, ticker, confidence = get_context_expansions(
            "gaming studio acquisition antitrust"
        )
        assert ticker == "MSFT"
        assert confidence >= THRESHOLD
        assert len(expansions) > 0

    def test_gaming_acquisition_alone_infers_msft(self):
        # Only the gaming+acquisition rule fires (no "antitrust" in query).
        expansions, ticker, confidence = get_context_expansions(
            "gaming studio acquisition"
        )
        assert ticker == "MSFT"
        assert confidence >= THRESHOLD

    def test_general_acquisition_no_ticker(self):
        # "acquisition" without "gaming" — no context rule fires.
        expansions, ticker, confidence = get_context_expansions(
            "general acquisition risks"
        )
        assert ticker is None
        assert confidence == 0.0

    def test_cloud_competition_no_ticker(self):
        # Generic query; no rule covers cloud/competition without specifics.
        expansions, ticker, confidence = get_context_expansions("cloud competition")
        assert ticker is None
        assert confidence == 0.0

    def test_digital_markets_act_infers_aapl(self):
        expansions, ticker, confidence = get_context_expansions(
            "risks from the eu digital markets act on app store"
        )
        assert ticker == "AAPL"
        assert confidence >= THRESHOLD

    def test_export_chip_china_infers_nvda(self):
        expansions, ticker, confidence = get_context_expansions(
            "export chip china restriction advanced"
        )
        assert ticker == "NVDA"
        assert confidence >= THRESHOLD

    def test_autonomous_regulatory_approval_infers_tsla(self):
        expansions, ticker, confidence = get_context_expansions(
            "autonomous regulatory approval software"
        )
        assert ticker == "TSLA"
        assert confidence >= THRESHOLD

    def test_below_threshold_no_hard_filter(self):
        # training+inference+workload fires at 0.70 — below threshold.
        # Expansions are still returned, but ticker should NOT be used as a hard filter.
        expansions, ticker, confidence = get_context_expansions(
            "training inference workload performance"
        )
        assert len(expansions) > 0          # soft signal still present
        assert confidence < THRESHOLD       # caller must not apply hard filter

    def test_conflict_returns_no_ticker(self):
        # gaming+acquisition → MSFT (0.85) AND export+chip+china → NVDA (0.85).
        # Two different tickers inferred → conflict → None.
        expansions, ticker, confidence = get_context_expansions(
            "gaming acquisition export chip china"
        )
        assert ticker is None
        assert confidence == 0.0
        # Expansion strings from both rules are still accumulated.
        assert len(expansions) > 0

    def test_multiple_rules_same_ticker_takes_max_confidence(self):
        # gaming+acquisition (0.85) + antitrust+gaming (0.90) both fire → MSFT.
        # Confidence should be max of the two = 0.90.
        _, ticker, confidence = get_context_expansions(
            "antitrust gaming acquisition studio"
        )
        assert ticker == "MSFT"
        assert confidence == pytest.approx(0.90)

    def test_expansions_contain_ticker_specific_strings(self):
        # Even when confidence is below threshold, expansion strings should
        # contain company-specific terms to bias the embedding search.
        expansions, _, _ = get_context_expansions("gaming studio acquisition antitrust")
        combined = " ".join(expansions).lower()
        assert "msft" in combined or "activision" in combined or "xbox" in combined


# ---------------------------------------------------------------------------
# QueryParser.parse() — integration tests (threshold gate + explicit override)
# ---------------------------------------------------------------------------

class TestQueryParserTickerInference:

    def setup_method(self):
        self.parser = QueryParser()

    # ---- Required test cases ----

    def test_gaming_studio_acquisition_antitrust_resolves_msft(self):
        """Query [36] scenario: no explicit ticker, high-confidence inference → MSFT hard filter."""
        pq = self.parser.parse("gaming studio acquisition antitrust")
        assert pq.tickers == ["MSFT"]

    def test_general_acquisition_risks_no_ticker(self):
        """Generic acquisition query → no context rule fires → tickers empty."""
        pq = self.parser.parse("general acquisition risks")
        assert pq.tickers == []

    def test_apple_supply_chain_explicit(self):
        """Explicit company name → extracted directly, independent of context expansion."""
        pq = self.parser.parse("Apple supply chain risks")
        assert "AAPL" in pq.tickers

    def test_cloud_competition_no_ticker(self):
        """Generic cloud/competition query → no ticker."""
        pq = self.parser.parse("cloud competition")
        assert pq.tickers == []

    # ---- Additional coverage ----

    def test_explicit_ticker_overrides_inferred(self):
        # "Microsoft gaming acquisition" — "microsoft" is extracted explicitly.
        # Explicit ticker should be used, not the inferred one (same result here,
        # but the path is different — testing that explicit wins regardless).
        pq = self.parser.parse("Microsoft gaming acquisition antitrust")
        assert pq.tickers == ["MSFT"]

    def test_below_threshold_confidence_no_hard_filter(self):
        # training+inference+workload → NVDA at 0.70 → NOT applied as hard filter.
        pq = self.parser.parse("training inference workload performance")
        assert pq.tickers == []

    def test_inferred_ticker_present_in_expansion_variants(self):
        # Even when no explicit ticker in query, MSFT-specific strings must
        # appear in expanded_queries so the embedding search is biased correctly.
        pq = self.parser.parse("gaming studio acquisition antitrust")
        combined = " ".join(pq.expanded_queries).lower()
        assert "msft" in combined or "activision" in combined or "xbox" in combined

    def test_two_explicit_tickers_both_preserved(self):
        # Comparison query — both tickers must survive.
        pq = self.parser.parse("Compare Apple and Microsoft cloud risk disclosures")
        assert "AAPL" in pq.tickers
        assert "MSFT" in pq.tickers
