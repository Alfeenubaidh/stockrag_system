from __future__ import annotations

from typing import Union

# Thresholds
_RETRIEVAL_THRESHOLD = 0.6
_GENERATION_THRESHOLD = 0.6
_E2E_THRESHOLD = 0.6


def _clamp(value: Union[int, float]) -> float:
    """
    Clamp value into [0.0, 1.0] range and cast to float.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, v))


def score_e2e(R: float, G: float) -> dict[str, float | str]:
    """
    Compute end-to-end score and classify failure mode.

    Args:
        R: Retrieval composite score (expected 0–1).
        G: Generation composite score (expected 0–1).

    Returns:
        {
            "R": float,
            "G": float,
            "E2E": float,
            "failure_type": str
        }
    """

    # --- sanitize inputs ---
    R = _clamp(R)
    G = _clamp(G)

    # --- compute E2E ---
    e2e = R * G

    # --- failure classification ---
    retrieval_fail = R < _RETRIEVAL_THRESHOLD
    generation_fail = G < _GENERATION_THRESHOLD

    if retrieval_fail and generation_fail:
        failure_type = "both"
    elif retrieval_fail:
        failure_type = "retrieval"
    elif generation_fail:
        failure_type = "generation"
    elif e2e < _E2E_THRESHOLD:
        # both individually pass but combined signal is weak
        failure_type = "both"
    else:
        failure_type = "pass"

    return {
        "R": round(R, 4),
        "G": round(G, 4),
        "E2E": round(e2e, 4),
        "failure_type": failure_type,
    }