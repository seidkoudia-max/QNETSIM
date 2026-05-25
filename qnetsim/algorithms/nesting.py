from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2, prod
from typing import Iterable, List, Sequence

from channels.fso import werner_to_fidelity
from quantum.memory import MemoryModelBase


@dataclass(frozen=True)
class NestingSegment:
    link_werners: List[float]
    round_durations_s: List[float]


def binary_nesting_bsm_counts(n_repeaters: int) -> List[int]:
    counts: List[int] = []
    remaining = max(0, int(n_repeaters))
    while remaining > 0:
        n_round = ceil(remaining / 2)
        counts.append(n_round)
        remaining -= n_round
    return counts


def binary_nesting_round_durations(
    n_repeaters: int, round_duration_s: float
) -> List[float]:
    return [float(round_duration_s)] * len(binary_nesting_bsm_counts(n_repeaters))


def _memory_factor(memory: MemoryModelBase, wait_s: float) -> float:
    return max(0.0, min(1.0, float(memory.coherence_factor(wait_s))))


def nested_werner(
    *,
    link_werners: Sequence[float],
    memory: MemoryModelBase,
    round_durations_s: Sequence[float],
) -> float:
    if not link_werners:
        return 0.0
    n_repeaters = max(0, len(link_werners) - 1)
    round_counts = binary_nesting_bsm_counts(n_repeaters)
    prefixes: List[float] = []
    total = 0.0
    for dt in round_durations_s:
        total += max(0.0, float(dt))
        prefixes.append(total)

    w_links = prod(max(0.0, min(1.0, float(w))) for w in link_werners)
    w_mem = 1.0
    for idx, n_bsms in enumerate(round_counts):
        if idx >= len(prefixes):
            break
        w_mem *= _memory_factor(memory, prefixes[idx]) ** (2 * int(n_bsms))
    if prefixes:
        w_mem *= _memory_factor(memory, prefixes[-1]) ** 2
    return max(0.0, min(1.0, w_links * w_mem))


def late_nesting_werner(
    *,
    link_werners: Sequence[float],
    memory: MemoryModelBase,
    round_durations_s: Sequence[float],
    extra_wait_s: float,
) -> float:
    base = nested_werner(
        link_werners=link_werners,
        memory=memory,
        round_durations_s=round_durations_s,
    )
    extra = _memory_factor(memory, max(0.0, float(extra_wait_s)))
    return max(0.0, min(1.0, base * extra))


def segmented_nesting_werner(
    *,
    segments: Iterable[NestingSegment],
    memory: MemoryModelBase,
    transition_waits_s: Sequence[float] = (),
) -> float:
    total = 1.0
    for seg in segments:
        total *= nested_werner(
            link_werners=seg.link_werners,
            memory=memory,
            round_durations_s=seg.round_durations_s,
        )
    for wait_s in transition_waits_s:
        total *= _memory_factor(memory, wait_s)
    return max(0.0, min(1.0, total))


def nested_fidelity(
    *,
    link_werners: Sequence[float],
    memory: MemoryModelBase,
    round_durations_s: Sequence[float],
) -> float:
    return werner_to_fidelity(
        nested_werner(
            link_werners=link_werners,
            memory=memory,
            round_durations_s=round_durations_s,
        )
    )


__all__ = [
    "NestingSegment",
    "binary_nesting_bsm_counts",
    "binary_nesting_round_durations",
    "late_nesting_werner",
    "nested_fidelity",
    "nested_werner",
    "segmented_nesting_werner",
]
