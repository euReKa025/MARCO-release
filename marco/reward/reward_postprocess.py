from __future__ import annotations


def post_process_rewards(args, samples):
    """Optional hook placeholder for future reward post-process.

    Currently returns identity mapping.
    """
    raw = [sample.reward for sample in samples]
    return raw, raw
