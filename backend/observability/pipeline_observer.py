from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TRACES_DIR = _PROJECT_ROOT / "data" / "logs" / "traces"


# ---------------------------------------------------------------------------
# Trace dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ParsingTrace:
    original_query: str
    expanded_queries: list[str]
    inferred_ticker: Optional[str]
    confidence: float
    section_hint: Optional[str]
    active_tickers: list[str]


@dataclass
class RetrievalHit:
    chunk_id: str
    ticker: Optional[str]
    section: str
    score: float
    text_preview: str  # first 120 chars


@dataclass
class RetrievalTrace:
    queries_sent: list[str]
    filter_applied: Optional[str]
    hits: list[RetrievalHit]


@dataclass
class RerankingHit:
    chunk_id: str
    ticker: Optional[str]
    section: str
    ce_score: float
    section_bonus: float
    keyword_bonus: float
    boilerplate_penalty: float
    final_score: float


@dataclass
class RerankingTrace:
    section_hint: Optional[str]
    hits: list[RerankingHit]


@dataclass
class GenerationTrace:
    prompt_length: int
    raw_output_length: int
    corrected_output_length: int
    citations_added: int
    citations_removed: int
    answer_tickers: list[str]
    raw_output: str = ""
    corrected_output: str = ""


@dataclass
class EvalScores:
    r_score: float
    g_score: float
    e2e_score: float
    failure_type: str       # "pass" | "retrieval" | "generation" | "both" | "generation_crash"
    generation_crashed: bool


@dataclass
class QueryTrace:
    query_id: str
    query_text: str
    run_id: str
    started_at: str
    finished_at: Optional[str] = None
    status: str = "in_progress"
    parsing: Optional[ParsingTrace] = None
    retrieval: Optional[RetrievalTrace] = None
    reranking: Optional[RerankingTrace] = None
    generation: Optional[GenerationTrace] = None
    eval_scores: Optional[EvalScores] = None


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------

class PipelineObserver:
    def __init__(self) -> None:
        self._run_id: str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self._current: Optional[QueryTrace] = None

    def start_trace(self, query_id: str, query_text: str) -> None:
        self._current = QueryTrace(
            query_id=query_id,
            query_text=query_text,
            run_id=self._run_id,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

    def _require_trace(self) -> QueryTrace | None:
        return self._current

    def log_parsing(
        self,
        original_query: str,
        expanded_queries: list[str],
        inferred_ticker: Optional[str],
        confidence: float,
        section_hint: Optional[str],
        active_tickers: list[str],
    ) -> None:
        t = self._require_trace()
        if t is None:
            return
        t.parsing = ParsingTrace(
            original_query=original_query,
            expanded_queries=expanded_queries or [],
            inferred_ticker=inferred_ticker,
            confidence=confidence,
            section_hint=section_hint,
            active_tickers=active_tickers,
        )

    def log_retrieval(
        self,
        queries_sent: list[str],
        filter_applied: Optional[str],
        results: list[Any],
    ) -> None:
        t = self._require_trace()
        if t is None:
            return
        hits = [
            RetrievalHit(
                chunk_id=r.chunk_id,
                ticker=r.metadata.get("ticker"),
                section=r.section,
                score=round(r.score, 4),
                text_preview=r.text[:120],
            )
            for r in results
        ]
        t.retrieval = RetrievalTrace(
            queries_sent=queries_sent,
            filter_applied=filter_applied,
            hits=hits,
        )

    def log_reranking(
        self,
        section_hint: Optional[str],
        hits: list[dict[str, Any]],
    ) -> None:
        t = self._require_trace()
        if t is None:
            return
        rerank_hits = [
            RerankingHit(
                chunk_id=h["chunk_id"],
                ticker=h.get("ticker"),
                section=h.get("section", ""),
                ce_score=round(h.get("ce_score", 0.0), 4),
                section_bonus=round(h.get("section_bonus", 0.0), 4),
                keyword_bonus=round(h.get("keyword_bonus", 0.0), 4),
                boilerplate_penalty=round(h.get("boilerplate_penalty", 0.0), 4),
                final_score=round(h.get("final_score", 0.0), 4),
            )
            for h in hits
        ]
        t.reranking = RerankingTrace(section_hint=section_hint, hits=rerank_hits)

    def log_generation(
        self,
        prompt: str,
        raw_output: str,
        corrected_output: str,
        citations_added: int,
        citations_removed: int,
        answer_tickers: list[str],
    ) -> None:
        t = self._require_trace()
        if t is None:
            return
        t.generation = GenerationTrace(
            prompt_length=len(prompt),
            raw_output_length=len(raw_output),
            corrected_output_length=len(corrected_output),
            citations_added=citations_added,
            citations_removed=citations_removed,
            answer_tickers=answer_tickers,
        )
        t.generation.raw_output = raw_output
        t.generation.corrected_output = corrected_output

    def log_eval_scores(
        self,
        r_score: float,
        g_score: float,
        e2e_score: float,
        failure_type: str,
        generation_crashed: bool,
    ) -> None:
        t = self._require_trace()
        if t is None:
            return
        t.eval_scores = EvalScores(
            r_score=round(r_score, 4),
            g_score=round(g_score, 4),
            e2e_score=round(e2e_score, 4),
            failure_type=failure_type,
            generation_crashed=generation_crashed,
        )

    def flush_trace(self, status: str = "ok") -> None:
        t = self._current
        if t is None:
            logger.warning("flush_trace called with no active trace")
            return
        t.finished_at = datetime.now(timezone.utc).isoformat()
        t.status = status

        out_dir = _TRACES_DIR / t.run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{t.query_id}.json"

        try:
            out_path.write_text(
                json.dumps(asdict(t), indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.error("Failed to write trace %s: %s", out_path, exc)
        finally:
            self._current = None


# Module-level singleton
observer = PipelineObserver()
