from __future__ import annotations

import sys
import unittest
from pathlib import Path


BUNDLE_APP_ROOT = Path(__file__).resolve().parents[1] / "qnetsim"
if str(BUNDLE_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(BUNDLE_APP_ROOT))

from framework.core.des import DESEnv, Entity, EventHandler
from framework.protocols.protocol import Protocol


class RecorderEntity(Entity):
    def __init__(self, name: str, env: DESEnv, sink: list[tuple[str, int]]) -> None:
        super().__init__(name, env)
        self._sink = sink

    def mark(self, label: str) -> None:
        self._sink.append((label, int(self.env.now)))


class RecorderProtocol(Protocol):
    def __init__(self, name: str, env: DESEnv, sink: list[tuple[str, int]]) -> None:
        super().__init__(name, env)
        self._sink = sink

    def mark(self, label: str) -> None:
        self._sink.append((label, int(self.env.now)))


class FrameworkCoreTests(unittest.TestCase):
    def test_default_environment_is_set_and_retrieved(self) -> None:
        env = DESEnv(name="default-env-check", default=True)
        self.assertIs(DESEnv.get_default_env(), env)

    def test_schedule_now_after_and_at_follow_time_order(self) -> None:
        env = DESEnv(name="time-order")
        sink: list[tuple[str, int]] = []
        entity = RecorderEntity("entity", env, sink)

        entity.scheduler.schedule_after(20, EventHandler(entity, "mark", ["after"]))
        entity.scheduler.schedule_at(10, EventHandler(entity, "mark", ["at"]))
        entity.scheduler.schedule_now(EventHandler(entity, "mark", ["now"]))

        env.run()

        self.assertEqual(
            sink,
            [("now", 0), ("at", 10), ("after", 20)],
        )

    def test_priority_breaks_ties_at_same_time(self) -> None:
        env = DESEnv(name="priority-order")
        sink: list[tuple[str, int]] = []
        entity = RecorderEntity("entity", env, sink)

        entity.scheduler.schedule_at(100, EventHandler(entity, "mark", ["priority-5"]), priority=5)
        entity.scheduler.schedule_at(100, EventHandler(entity, "mark", ["priority-1"]), priority=1)
        entity.scheduler.schedule_at(100, EventHandler(entity, "mark", ["priority-3"]), priority=3)

        env.run()

        self.assertEqual(
            [label for label, _ in sink],
            ["priority-1", "priority-3", "priority-5"],
        )

    def test_cancelled_events_do_not_fire(self) -> None:
        env = DESEnv(name="cancel-order")
        sink: list[tuple[str, int]] = []
        entity = RecorderEntity("entity", env, sink)

        event = entity.scheduler.schedule_after(10, EventHandler(entity, "mark", ["cancelled"]))
        entity.scheduler.schedule_after(20, EventHandler(entity, "mark", ["kept"]))
        entity.scheduler.cancel(event)

        env.run()

        self.assertEqual(sink, [("kept", 20)])

    def test_protocol_owned_scheduler_submits_events(self) -> None:
        env = DESEnv(name="protocol-order")
        sink: list[tuple[str, int]] = []
        protocol = RecorderProtocol("proto", env, sink)

        protocol.scheduler.schedule_after(30, EventHandler(protocol, "mark", ["proto-event"]))
        env.run()

        self.assertEqual(sink, [("proto-event", 30)])
        self.assertEqual(len(protocol.agenda), 1)
        self.assertEqual(len(protocol.signed_events), 1)


if __name__ == "__main__":
    unittest.main()
