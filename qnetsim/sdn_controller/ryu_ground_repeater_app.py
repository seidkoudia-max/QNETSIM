from __future__ import annotations

import json
from typing import Any, Dict

from ryu.base import app_manager
from ryu.app.wsgi import ControllerBase, WSGIApplication, route
from webob import Response

from sdn_controller.ground_repeater_controller_core import GroundRepeaterSdnControllerCore


def _json_response(payload: Any, status: int = 200) -> Response:
    body = json.dumps(payload).encode("utf-8")
    return Response(status=status, content_type="application/json", body=body)


class GroundRepeaterSdnRestController(ControllerBase):
    def __init__(self, req, link, data, **config):
        super().__init__(req, link, data, **config)
        self.app: "GroundRepeaterSdnRyuApp" = data["app"]

    @route("sdn", "/sdn/event", methods=["POST"])
    def sdn_event(self, req, **kwargs):
        try:
            payload = req.json if hasattr(req, "json") else json.loads(req.body.decode("utf-8"))
        except Exception as e:
            return _json_response({"error": f"invalid json: {e}"}, status=400)
        return _json_response(self.app.handle_event(payload))

    @route("sdn", "/sdn/fso/update", methods=["POST"])
    def sdn_fso_update(self, req, **kwargs):
        try:
            payload = req.json if hasattr(req, "json") else json.loads(req.body.decode("utf-8"))
        except Exception as e:
            return _json_response({"error": f"invalid json: {e}"}, status=400)
        self.app.controller.update_link_state(payload)
        return _json_response({"ok": True})


class GroundRepeaterSdnRyuApp(app_manager.RyuApp):
    _CONTEXTS = {"wsgi": WSGIApplication}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        wsgi = kwargs["wsgi"]
        wsgi.register(GroundRepeaterSdnRestController, {"app": self})
        self.controller = self._build_controller(max_modes_per_label=256, max_bsm_per_cycle=4096)

    @staticmethod
    def _build_controller(*, max_modes_per_label: int, max_bsm_per_cycle: int) -> GroundRepeaterSdnControllerCore:
        return GroundRepeaterSdnControllerCore(
            max_modes_per_label=max_modes_per_label,
            max_bsm_per_cycle=max_bsm_per_cycle,
        )

    def handle_event(self, payload: Dict[str, Any]) -> Any:
        et = str(payload.get("type", "")).upper()
        if et == "RESET":
            self.controller = self._build_controller(
                max_modes_per_label=int(payload.get("max_modes_per_label", 256)),
                max_bsm_per_cycle=int(payload.get("max_bsm_per_cycle", 4096)),
            )
            self.controller.reset()
            return {"ok": True}
        if et == "ENT_REQUEST":
            self.controller.submit_request(
                req_id=int(payload["req_id"]),
                src=str(payload["src"]),
                dst=str(payload["dst"]),
                f_min=float(payload.get("f_min", 0.25)),
                timeout_ps=int(payload.get("timeout_ps", 0)),
                t_ps=int(payload.get("t_ps", 0)),
                target_pairs=int(payload.get("target_pairs", 0)),
            )
            return {"ok": True}
        if et == "POLL":
            return self.controller.poll(
                str(payload.get("node", "")),
                int(payload.get("t_ps", 0)),
            )
        if et == "REPORT_STORAGE":
            self.controller.report_storage(
                req_id=int(payload["req_id"]),
                label=str(payload["label"]),
                accepted_count=int(payload.get("accepted_count", 0)),
            )
            return {"ok": True}
        if et == "SWAP_RESULT":
            self.controller.report_swap_result(
                req_id=int(payload["req_id"]),
                success_count=int(payload.get("success_count", 0)),
            )
            return {"ok": True}
        return {"ok": False, "error": f"unsupported event type: {et}"}
