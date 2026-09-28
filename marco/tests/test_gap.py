import pytest

from marco.reward.gap import gap_progress_reward, similarity_penalty


def test_gap_progress_positive_when_gap_reduces():
    assert gap_progress_reward(1.0, 0.4, 0.5) == 0.5


def test_gap_progress_negative_when_gap_worsens():
    assert gap_progress_reward(0.2, 0.7, 0.5) == pytest.approx(-0.5)


def test_similarity_penalty():
    assert similarity_penalty(0.8, 0.5, 0.5) == 0.0
    assert similarity_penalty(0.2, 0.5, 0.5) < 0.0
