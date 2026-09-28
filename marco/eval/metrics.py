from __future__ import annotations


def success_rate(flags: list[bool]) -> float:
    if not flags:
        return 0.0
    return sum(1 for f in flags if f) / len(flags)
