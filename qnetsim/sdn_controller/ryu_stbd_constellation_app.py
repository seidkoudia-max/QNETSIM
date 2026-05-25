# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from typing import Any, Dict

from ryu.base import app_manager
from ryu.app.wsgi import ControllerBase, WSGIApplication, route
from webob import Response

from sdn_controller.stbd_segment_controller_core import STBDSegmentControllerCore


def _json_response(payload: Any, status: int = 200) -> Response:
    body = json.dumps(payload).encode("utf-8")
    return Response(status=status, content_type="application/json", body=body)


class STBDConstellationRestController(ControllerBase):
    def __init__(self, req, link, data, **config):
        super().__init__(req, link, data, **config)
        self.app: "STBDConstellationRyuApp" = data["app"]

    @route("sdn", "/sdn/event", methods=["POST"])
    def sdn_event(self, req, **kwargs):
        try:
            payload = req.json if hasattr(req, "json") else json.loads(req.body.decode("utf-8"))
        except Exception as exc:
            return _json_response({"error": f"invalid json: {exc}"}, status=400)

        response = self.app.handle_event(payload)
        return _json_response(response)


class STBDConstellationRyuApp(app_manager.RyuApp):
    _CONTEXTS = {"wsgi": WSGIApplication}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        wsgi = kwargs["wsgi"]
        wsgi.register(STBDConstellationRestController, {"app": self})
        self.core = STBDSegmentControllerCore()

    def handle_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        event_type = str(payload.get("type", "")).upper()
        if event_type == "RESET":
            self.core.reset()
            return {"ok": True}
        if event_type == "INSTALL_REQUEST":
            self.core.install_request(
                req_id=int(payload["req_id"]),
                execution_mode=str(payload.get("execution_mode", "basic")),
                final_label=str(payload.get("final_label", "")),
                route_start_s=float(payload.get("route_start_s", 0.0)),
                route_end_s=float(payload.get("route_end_s", 0.0)),
                final_segment_start_s=float(payload.get("final_segment_start_s", 0.0)),
                final_segment_end_s=float(payload.get("final_segment_end_s", 0.0)),
                segments=list(payload.get("segments", [])),
            )
            return {"ok": True}
        if event_type == "POLL":
            return self.core.poll(
                req_id=int(payload.get("req_id", 0)),
                t_s=float(payload.get("t_s", 0.0)),
            )
        return {"error": f"unsupported event type: {event_type}"}
