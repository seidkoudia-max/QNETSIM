from __future__ import annotations

import zlib


def stable_hash_u32(value: str) -> int:
    return zlib.crc32(str(value).encode("utf-8")) & 0xFFFFFFFF


__all__ = [
    "stable_hash_u32",
]
