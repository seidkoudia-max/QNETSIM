from __future__ import annotations

from io import StringIO
from typing import Iterable


def show_heap(heap: Iterable[object]) -> None:
    """Print a compact view of a DES heap-like queue."""

    output = StringIO()
    for idx, item in enumerate(heap):
        output.write(f"{idx}: {item}\n")
    print(output.getvalue().rstrip())


def is_heap(heap: list[object]) -> bool:
    """Best-effort check that a list obeys min-heap ordering."""

    for idx in range(1, len(heap)):
        parent = (idx - 1) // 2
        if heap[parent] > heap[idx]:
            return False
    return True

