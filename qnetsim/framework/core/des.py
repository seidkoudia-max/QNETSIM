from __future__ import annotations

import heapq
import itertools
import time
from abc import ABC
from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, Iterable, List, Optional, Sequence, Union

from .log import log_init

__all__ = ["Entity", "Event", "EventHandler", "Scheduler", "DESEnv"]


class EventHandler:
    """Reference to a method that should be executed when an event fires."""

    def __init__(self, owner: Union["Entity", "Protocol"], method: str, params: Optional[Sequence[Any]] = None, **kwargs):
        self.owner = owner
        self.method = method
        self.params = [] if params is None else list(params)
        self.kwargs = kwargs

    def handle(self) -> None:
        getattr(self.owner, self.method)(*self.params, **self.kwargs)

    def __repr__(self) -> str:
        return f"EventHandler(owner={getattr(self.owner, 'name', type(self.owner).__name__)}, method={self.method})"


@unique
class EventStatus(Enum):
    CREATED = "Created"
    SCHEDULED = "Scheduled"
    PROCESSED = "Processed"
    CANCELLED = "Cancelled"


@dataclass
class Event:
    time: int
    handler: EventHandler
    signature: Union["Entity", "Protocol"]
    priority: int = 0
    status: EventStatus = EventStatus.CREATED

    def cancel(self) -> None:
        self.status = EventStatus.CANCELLED

    @classmethod
    def events_to_dataframe(cls, events: Iterable["Event"]) -> list[dict[str, Any]]:
        return [
            {
                "time": event.time,
                "owner": getattr(event.handler.owner, "name", type(event.handler.owner).__name__),
                "method": event.handler.method,
                "signature": getattr(event.signature, "name", type(event.signature).__name__),
                "priority": event.priority,
                "status": event.status.value,
            }
            for event in events
        ]


class Scheduler:
    """Event scheduler attached to an entity or protocol owner."""

    def __init__(self, owner: Union["Entity", "Protocol"], env: Optional["DESEnv"] = None):
        self.owner = owner
        if env is None:
            env = getattr(owner, "env", None)
        if env is None:
            env = DESEnv.get_default_env()
        if env is None:
            raise ValueError("A DES environment is required to create a scheduler.")
        self.env = env

    def schedule_at(self, time_ps: int, handler: EventHandler, priority: Optional[int] = None) -> Event:
        if int(time_ps) < self.env.now:
            raise ValueError("Cannot schedule an event before the current DES time.")
        event = Event(
            time=int(time_ps),
            handler=handler,
            signature=self.owner,
            priority=0 if priority is None else int(priority),
        )
        event.status = EventStatus.SCHEDULED
        heapq.heappush(
            self.env._queue,
            (event.time, event.priority, next(self.env._counter), event),
        )
        if hasattr(handler.owner, "agenda"):
            handler.owner.agenda.append(event)
        if hasattr(self.owner, "signed_events"):
            self.owner.signed_events.append(event)
        self.env.future_events.append(event)
        return event

    def schedule_after(self, delay_ps: int, handler: EventHandler, priority: Optional[int] = None) -> Event:
        if delay_ps < 0:
            raise ValueError("Event delay must be nonnegative.")
        return self.schedule_at(self.env.now + int(delay_ps), handler, priority)

    def schedule_now(self, handler: EventHandler, priority: Optional[int] = None) -> Event:
        return self.schedule_at(self.env.now, handler, priority)

    def cancel(self, event: Event) -> None:
        event.cancel()


class DESEnv:
    """Simple DES environment for the standalone bundle."""

    __DEFAULT_ENV: Optional["DESEnv"] = None

    def __init__(self, name: str = "des", default: bool = False):
        self.name = name
        self.entities: list[Entity] = []
        self.future_events: list[Event] = []
        self._queue: list[tuple[int, int, int, Event]] = []
        self._counter = itertools.count()
        self._now = 0
        self.logger = log_init(f"qnetsim.framework.{self.name}")
        self.run_start_wallclock: Optional[float] = None
        self.run_stop_wallclock: Optional[float] = None
        if default:
            self.set_default_env()

    @property
    def now(self) -> int:
        return self._now

    def init(self) -> None:
        for entity in list(self.entities):
            entity.init()

    def set_default_env(self) -> None:
        DESEnv.__DEFAULT_ENV = self

    @classmethod
    def get_default_env(cls) -> Optional["DESEnv"]:
        return cls.__DEFAULT_ENV

    def run(self, end_time: Optional[int] = None, logging: bool = False, summary: bool = True) -> None:
        limit = float("inf") if end_time is None else int(end_time)
        self.run_start_wallclock = time.time()
        processed = 0
        while self._queue:
            event_time, _priority, _idx, event = heapq.heappop(self._queue)
            if event.status is EventStatus.CANCELLED:
                continue
            if event_time > limit:
                heapq.heappush(self._queue, (event_time, event.priority, _idx, event))
                break
            self._now = event_time
            event.status = EventStatus.PROCESSED
            processed += 1
            if logging:
                self.logger.debug("t=%s owner=%s method=%s", event.time, getattr(event.handler.owner, "name", "?"), event.handler.method)
            event.handler.handle()
        self.run_stop_wallclock = time.time()
        if summary and logging:
            elapsed = self.run_stop_wallclock - self.run_start_wallclock
            self.logger.info("Processed %d events in %.3fs", processed, elapsed)


class Entity(ABC):
    """Base class for DES entities."""

    def __init__(self, name: str, env: Optional[DESEnv] = None):
        self.name = name
        if env is None:
            env = DESEnv.get_default_env()
            if env is None:
                raise ValueError("Default DES environment is None.")
        self.env = env
        self.env.entities.append(self)
        self.owner: Entity = self
        self.components: list[Entity] = []
        self.scheduler = Scheduler(self, self.env)
        self.agenda: list[Event] = []
        self.signed_events: list[Event] = []

    def init(self) -> None:
        """Override in subclasses."""

    def install(self, entities: Union["Entity", Sequence["Entity"]]) -> None:
        if isinstance(entities, Entity):
            self._install_entity(entities)
            return
        for entity in entities:
            self._install_entity(entity)

    def _install_entity(self, other: "Entity") -> None:
        if self.env is not other.env:
            raise ValueError("Cannot install an entity with different environment.")
        if other in self.components:
            return
        other.owner = self
        self.components.append(other)

    def print_components(self) -> None:
        for idx, component in enumerate(self.components, start=1):
            print(f"{idx}. {component.__class__.__name__} {component.name}")

    def print_agenda(self) -> None:
        print(Event.events_to_dataframe(self.agenda))

    def print_signed_events(self) -> None:
        print(Event.events_to_dataframe(self.signed_events))


class Protocol:  # pragma: no cover - runtime compatibility surface
    """Forward declaration for type checking and protocol ownership."""

    name: str
    env: DESEnv
    scheduler: Scheduler
