from __future__ import annotations


def directional_progress_delta(prev_score: float, new_score: float) -> float:
    return float(new_score - prev_score)


def clip_directional_progress(improvement: float, clip_c: float) -> float:
    clip_c = max(float(clip_c), 1e-6)
    return min(max(float(improvement), -clip_c), clip_c)


def gap_progress_reward(prev_gap: float, new_gap: float, clip_delta: float) -> float:
    # Legacy helper kept for compatibility with old tests/scripts.
    delta = prev_gap - new_gap
    if delta > clip_delta:
        return clip_delta
    if delta < -clip_delta:
        return -clip_delta
    return delta


def similarity_penalty(similarity: float, threshold: float, lambda_sim: float) -> float:
    return -lambda_sim * max(0.0, threshold - similarity)
