from __future__ import annotations

from abc import ABC
from typing import Optional

from framework.core.des import DESEnv, Event, EventHandler, Scheduler

__all__ = ["Protocol", "SubProtocol", "ProtocolStack"]


class Protocol(ABC):
    """QNETSIM-native protocol compatibility layer."""

    def __init__(self, name: str, env: Optional[DESEnv] = None):
        self.name = name
        if env is None:
            env = DESEnv.get_default_env()
            if env is None:
                raise ValueError("Default DES environment is None.")
        self.env = env
        self.owner: ProtocolStack | Protocol = self
        self.node = None
        self.scheduler = Scheduler(self, self.env)
        self.agenda: list[Event] = []
        self.signed_events: list[Event] = []

    def init(self) -> None:
        """Override in subclasses if protocol state needs initialization."""


class SubProtocol(Protocol):
    def __init__(self, name: str, env: Optional[DESEnv] = None, parent: Optional["ProtocolStack"] = None):
        super().__init__(name=name, env=env)
        self.parent = parent
        if parent is not None:
            self.owner = parent
            self.node = parent.node


class ProtocolStack(Protocol):
    def __init__(self, name: str, env: Optional[DESEnv] = None, node=None):
        super().__init__(name=name, env=env)
        self.node = node
        self.protocols: list[Protocol] = []

    def add(self, protocol: Protocol) -> Protocol:
        protocol.owner = self
        protocol.node = self.node
        protocol.env = self.env
        protocol.scheduler = Scheduler(protocol, self.env)
        self.protocols.append(protocol)
        return protocol

    def init(self) -> None:
        for protocol in self.protocols:
            protocol.init()

