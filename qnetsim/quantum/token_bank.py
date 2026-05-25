from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from quantum.memory import MemoryModelBase, build_memory_model
from sim.seeding import stable_hash_u32


# ============================================================
# Token + multimode memory + token bank
# ============================================================
# Purpose:
# - Provide a concrete storage layer for entangled pairs ("tokens").
# - Support multimode memories with write success, recall probability,
#   and decoherence via the selected memory model.
# - Offer a per-node TokenBank that groups memories by label.


# ============================================================
# Data models
# ============================================================
@dataclass(frozen=True)
class MemoryParams:
    memory_id: str
    n_modes: int
    platform: str = "REID"
    rng_seed: int = 12345


@dataclass
class TokenRecord:
    req_id: int
    label: str
    token_id: str
    ready_ps: int
    werner_w_init: float
    fidelity_init: float


# ============================================================
# Multi-mode memory
# ============================================================
class MultiModeMemory:
    """
    True multimode memory.

    - Write success uses model.write_success_prob()
    - Recall uses conditional_recall_prob(t)
    - Decoherence applies coherence_factor(t) to Werner parameter
    """

    def __init__(self, params: MemoryParams):
        self.p = params
        self.model: MemoryModelBase = build_memory_model(params.platform)
        self._modes: List[Optional[TokenRecord]] = [None] * int(params.n_modes)
        self._rng = random.Random(
            (int(params.rng_seed) ^ stable_hash_u32(params.memory_id)) & 0xFFFFFFFF
        )

    # ------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------
    @property
    def memory_id(self) -> str:
        return self.p.memory_id

    @property
    def n_modes(self) -> int:
        return int(self.p.n_modes)

    # ------------------------------------------------------------
    # Management
    # ------------------------------------------------------------
    def clear_req(self, req_id: int) -> None:
        for i, rec in enumerate(self._modes):
            if rec is not None and int(rec.req_id) == int(req_id):
                self._modes[i] = None

    # ------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------
    def store_in_mode(self, mode: int, rec: TokenRecord) -> bool:
        m = int(mode)
        if m < 0 or m >= len(self._modes):
            return False
        if self._modes[m] is not None:
            return False

        p_write = max(0.0, min(1.0, float(self.model.write_success_prob())))
        if self._rng.random() >= p_write:
            return False

        self._modes[m] = rec
        return True

    def store_any(self, rec: TokenRecord) -> Tuple[bool, int]:
        for m in range(len(self._modes)):
            if self._modes[m] is None:
                ok = self.store_in_mode(m, rec)
                return ok, m if ok else -1
        return False, -1

    # ------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------
    def get(self, token_id: str) -> Tuple[Optional[TokenRecord], Optional[int]]:
        for i, rec in enumerate(self._modes):
            if rec is not None and rec.token_id == token_id:
                return rec, i
        return None, None

    def pop(self, token_id: str) -> Optional[TokenRecord]:
        for i, rec in enumerate(self._modes):
            if rec is not None and rec.token_id == token_id:
                self._modes[i] = None
                return rec
        return None

    # ------------------------------------------------------------
    # Recall + decoherence
    # ------------------------------------------------------------
    def recall_and_decohere(
        self, rec: TokenRecord, *, now_ps: int
    ) -> Tuple[bool, float, float, float]:
        """
        Returns:
            (ok, werner_w_out, fidelity_out, dt_s)
        """
        dt_s = max(0.0, float(int(now_ps) - int(rec.ready_ps)) / 1e12)

        p_rec = max(0.0, min(1.0, float(self.model.conditional_recall_prob(dt_s))))
        if self._rng.random() >= p_rec:
            return False, 0.0, 0.25, dt_s

        lam = max(0.0, min(1.0, float(self.model.coherence_factor(dt_s))))
        w_out = max(0.0, min(1.0, float(rec.werner_w_init) * lam))
        f_out = (1.0 + 3.0 * w_out) / 4.0

        return True, w_out, f_out, dt_s


# ============================================================
# Token bank (per node)
# ============================================================
@dataclass(frozen=True)
class TokenBankParams:
    n_modes_default: int = 8
    platform_default: str = "REID"
    rng_seed: int = 12345


class TokenBank:
    """
    Per-node dynamic memory bank: label -> MultiModeMemory
    """

    def __init__(self, *, node_name: str, params: TokenBankParams):
        self.node_name = node_name
        self.params = params
        self.memories_by_label: Dict[str, MultiModeMemory] = {}

    # ------------------------------------------------------------
    # Memory access
    # ------------------------------------------------------------
    def get_memory(self, label: str) -> MultiModeMemory:
        label = str(label)
        if label not in self.memories_by_label:
            mem_id = f"{self.node_name}:{label}:MM"
            self.memories_by_label[label] = MultiModeMemory(
                MemoryParams(
                    memory_id=mem_id,
                    n_modes=self.params.n_modes_default,
                    platform=self.params.platform_default,
                    rng_seed=self.params.rng_seed,
                )
            )
        return self.memories_by_label[label]

    # ------------------------------------------------------------
    # Request cleanup
    # ------------------------------------------------------------
    def clear_req(self, req_id: int) -> None:
        for m in self.memories_by_label.values():
            m.clear_req(req_id)

    # ------------------------------------------------------------
    # Token lookup
    # ------------------------------------------------------------
    def find_token(
        self, token_id: str
    ) -> Tuple[Optional[TokenRecord], Optional[MultiModeMemory]]:
        for m in self.memories_by_label.values():
            rec, _ = m.get(token_id)
            if rec is not None:
                return rec, m
        return None, None


__all__ = [
    "MemoryParams",
    "MultiModeMemory",
    "TokenBank",
    "TokenBankParams",
    "TokenRecord",
]
