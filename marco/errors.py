class MarcoError(Exception):
    """Base error for MARCO RL modules."""


class InvalidEvaluationError(MarcoError):
    """Candidate molecule is parseable but cannot be evaluated."""


class PredictorServiceError(MarcoError):
    """Property service call failed due to external service/network issues."""
