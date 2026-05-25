from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import json
import math
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Sequence
import urllib.error
import urllib.request

MPL_DIR = Path(__file__).resolve().parent / ".mplconfig"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
MPL_DIR.mkdir(exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from channels.fiber import FiberParams
from channels.fso import FSOParams
from des_ground_repeater_simulator import (
    KIRCHBERG_END_DEFAULT,
    PS_PER_S,
    WINDHOF_OGS_DEFAULT,
    build_parser,
    effective_fiber_length_km,
    default_satellites,
    ground_repeater_sites,
    require_qnet_des,
    representative_epoch_unix_s,
    run_once,
    select_satellite_for_ground_run,
    visibility_start_offset,
    write_outputs,
)
from sdn_controller.ground_repeater_controller_core import GroundRepeaterSdnControllerCore
from mobility.geometry import GeometryEngine
from mobility.propagator import GroundStation, SkyfieldPropagator


GALLERY_MANIFEST = "gallery_manifest.json"


@dataclass(frozen=True)
class GroundRunResult:
    out_dir: Path
    artifacts: Dict[str, Path]
    summary_path: Path
    metrics_path: Path
    selected_pass_path: Path


def default_ground_out_root() -> Path:
    return Path(__file__).resolve().parent / "Output" / "gui_ground_runs"


def save_gallery_manifest(result: GroundRunResult) -> None:
    image_items = []
    for key, path in result.artifacts.items():
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        image_items.append(
            {
                "title": key.replace("_", " ").title(),
                "path": str(path.resolve()),
            }
        )
    payload = {
        "out_dir": str(result.out_dir.resolve()),
        "images": image_items,
        "files": {
            "summary_txt": str(result.summary_path.resolve()),
            "metrics_csv": str(result.metrics_path.resolve()),
            "selected_pass_json": str(result.selected_pass_path.resolve()),
            **{key: str(path.resolve()) for key, path in result.artifacts.items()},
        },
    }
    (result.out_dir / GALLERY_MANIFEST).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_gallery_manifest(out_dir: Path) -> dict | None:
    manifest = out_dir / GALLERY_MANIFEST
    if not manifest.exists():
        return None
    return json.loads(manifest.read_text(encoding="utf-8"))


def _build_summary_text(*, sim, args) -> str:
    success_count = int(sim.cumulative_successes)
    nonzero_bins = sum(1 for row in sim.metrics if row.successes > 0)
    mean_fidelity = (
        sum(row.mean_fidelity_success * row.successes for row in sim.metrics if row.successes > 0) / success_count
        if success_count > 0
        else 0.0
    )
    peak_rate = max((row.throughput_pairs_per_s for row in sim.metrics), default=0.0)
    sat_visible_fraction = (
        sum(row.sat_visible for row in sim.metrics) / max(1, len(sim.metrics))
        if sim.metrics
        else 0.0
    )
    return "\n".join(
        [
            "QNETSIM Ground Scenario Summary",
            "",
            "Scenario:",
            f"  OGS repeater at {args.ogs_name}",
            "  satellite end node linked to the OGS via FSO",
            f"  ground end node at {args.ground_end_name} linked to the OGS via fiber",
            f"  controller backend: {str(getattr(args, 'ground_controller_backend', 'des_only'))}",
            "",
            "Key controls:",
            f"  duration_s: {float(args.duration_s):.3f}",
            f"  source_rate_hz: {float(args.source_rate_hz):.3f}",
            f"  min_elevation_deg: {float(args.min_elevation_deg):.3f}",
            f"  fiber_length_km_effective: {effective_fiber_length_km(args):.3f}",
            f"  mem_platform: {args.mem_platform}",
            f"  mem_t2_s: {float(args.mem_t2):.3f}",
            f"  max_modes_per_label: {int(args.max_modes_per_label)}",
            f"  max_bsm_per_cycle_effective: {int(args.max_bsm_per_cycle)}",
            "",
            "Metrics:",
            f"  cumulative_successes: {success_count}",
            f"  weighted_mean_fidelity: {mean_fidelity:.6f}",
            f"  peak_throughput_pairs_per_s: {peak_rate:.6f}",
            f"  nonzero_metric_bins: {nonzero_bins}",
            f"  total_metric_bins: {len(sim.metrics)}",
            f"  satellite_visibility_fraction: {sat_visible_fraction:.6f}",
            "",
            "Selected pass:",
            f"  original_satellite_name: {getattr(sim, 'selected_satellite_name', 'unknown')}",
        ]
    )


def _configure_ground_args(
    *,
    duration_s: float,
    metrics_dt_s: float,
    link_update_dt_s: float,
    gen_delay_s: float,
    swap_dt_s: float,
    source_rate_hz: float,
    max_modes_per_label: int,
    max_bsm_per_cycle: int,
    f_min: float,
    min_elevation_deg: float,
    visibility_scan_s: float,
    visibility_scan_step_s: float,
    visibility_prebuffer_s: float,
    pass_selection: str,
    pass_candidate_limit: int,
    pass_prefilter_step_s: float,
    force_visible: bool,
    fso_params: FSOParams,
    fiber_params: FiberParams,
    fiber_length_km: float,
    ogs_name: str,
    ogs_lat_deg: float,
    ogs_lon_deg: float,
    ogs_alt_m: float,
    ground_end_name: str,
    ground_end_lat_deg: float,
    ground_end_lon_deg: float,
    ground_end_alt_m: float,
    auto_fiber_length_from_sites: bool,
    mem_platform: str,
    mem_t2: float,
    mem_d: float,
    mem_t_mem: float,
    mem_n: float,
    mem_m: float,
    mem_eta_mode0: float,
    mem_eta_ti: float,
    bsm_v_hom: float,
    bsm_eta_det: float,
    bsm_sigma_det_s: float,
    bsm_delta_t_s: float,
    bsm_p_dc: float,
    bsm_swap_model: str,
    seed: int,
) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args([])
    args.seed = int(seed)
    args.duration_s = float(duration_s)
    args.metrics_dt_s = float(metrics_dt_s)
    args.link_update_dt_s = float(link_update_dt_s)
    args.gen_delay_s = float(gen_delay_s)
    args.swap_dt_s = float(swap_dt_s)
    args.source_rate_hz = float(source_rate_hz)
    args.max_modes_per_label = int(max_modes_per_label)
    # Keep end-to-end release bounded by the available memory modes.
    args.max_bsm_per_cycle = min(int(max_bsm_per_cycle), int(max_modes_per_label))
    args.f_min = float(f_min)
    args.min_elevation_deg = float(min_elevation_deg)
    args.visibility_scan_s = float(visibility_scan_s)
    args.visibility_scan_step_s = float(visibility_scan_step_s)
    args.visibility_prebuffer_s = float(visibility_prebuffer_s)
    args.pass_selection = str(pass_selection)
    args.pass_candidate_limit = int(pass_candidate_limit)
    args.pass_prefilter_step_s = float(pass_prefilter_step_s)
    args.force_visible = bool(force_visible)
    args.ogs_name = str(ogs_name)
    args.ogs_lat_deg = float(ogs_lat_deg)
    args.ogs_lon_deg = float(ogs_lon_deg)
    args.ogs_alt_m = float(ogs_alt_m)
    args.ground_end_name = str(ground_end_name)
    args.ground_end_lat_deg = float(ground_end_lat_deg)
    args.ground_end_lon_deg = float(ground_end_lon_deg)
    args.ground_end_alt_m = float(ground_end_alt_m)
    args.auto_fiber_length_from_sites = bool(auto_fiber_length_from_sites)

    args.fso_eta_src = float(fso_params.eta_src)
    args.fso_eta_det = float(fso_params.eta_det)
    args.fso_eta_cpl = float(fso_params.eta_cpl)
    args.fso_eta_point = float(fso_params.eta_point_static)
    args.fso_d_t = float(fso_params.d_t)
    args.fso_d_r = float(fso_params.d_r)
    args.fso_theta_div = float(fso_params.theta_div)
    args.fso_wavelength_m = float(fso_params.wavelength_m)
    args.fso_t_abs = float(fso_params.t_abs)
    args.fso_t_sca = float(fso_params.t_sca)
    args.fso_c_atm = float(fso_params.c_atm)
    args.fso_sigma_r2 = float(fso_params.sigma_r2)
    args.fso_ao_gain = float(fso_params.ao_gain)
    args.fso_turbulence_loss_coeff = float(fso_params.turbulence_loss_coeff)
    args.fso_turbulence_quality_coeff = float(fso_params.turbulence_quality_coeff)
    args.fso_delta_t = float(fso_params.delta_t)
    args.fso_n0 = float(fso_params.n0)
    args.fso_n_dc = float(fso_params.n_dc)
    args.fso_n_conv = float(fso_params.n_conv)
    args.fso_p_t = float(fso_params.p_t)
    args.fso_pointing_jitter_rad = float(fso_params.pointing_jitter_rad)
    args.disable_fso_pointing_jitter = not bool(fso_params.enable_pointing_jitter)
    args.disable_fso_stochastic_fading = not bool(fso_params.enable_stochastic_fading)

    args.fiber_length_km = float(fiber_length_km)
    args.fiber_loss_db_per_km = float(fiber_params.loss_db_per_km)
    args.fiber_insertion_loss_db = float(fiber_params.insertion_loss_db)
    args.fiber_eta_src = float(fiber_params.eta_src)
    args.fiber_eta_det = float(fiber_params.eta_det)
    args.fiber_eta_cpl = float(fiber_params.eta_cpl)
    args.fiber_eta_conn = float(fiber_params.eta_conn)
    args.fiber_polarization_error_rad = float(fiber_params.polarization_error_rad)
    args.fiber_phase_noise_rad = float(fiber_params.phase_noise_rad)
    args.fiber_temporal_mismatch_s = float(fiber_params.temporal_mismatch_s)
    args.fiber_coherence_time_s = float(fiber_params.coherence_time_s)
    args.fiber_p_t = float(fiber_params.p_t)
    args.fiber_n0 = float(fiber_params.n0)
    args.fiber_n_dc = float(fiber_params.n_dc)
    args.fiber_n_conv = float(fiber_params.n_conv)
    args.fiber_delta_t = float(fiber_params.delta_t)
    args.fiber_enable_length_fluctuations = bool(fiber_params.enable_length_fluctuations)
    args.fiber_length_jitter_fraction = float(fiber_params.length_jitter_fraction)
    args.fiber_max_snr_for_mapping = float(fiber_params.max_snr_for_mapping)

    args.mem_platform = str(mem_platform)
    args.mem_t2 = float(mem_t2)
    args.mem_d = float(mem_d)
    args.mem_t_mem = float(mem_t_mem)
    args.mem_n = float(mem_n)
    args.mem_m = float(mem_m)
    args.mem_eta_mode0 = float(mem_eta_mode0)
    args.mem_eta_ti = float(mem_eta_ti)

    args.bsm_v_hom = float(bsm_v_hom)
    args.bsm_eta_det = float(bsm_eta_det)
    args.bsm_sigma_det_s = float(bsm_sigma_det_s)
    args.bsm_delta_t_s = float(bsm_delta_t_s)
    args.bsm_p_dc = float(bsm_p_dc)
    args.bsm_swap_model = str(bsm_swap_model)
    return args


def _pick_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_local_port(port: int, *, timeout_s: float = 12.0) -> None:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.25)
            try:
                sock.connect(("127.0.0.1", int(port)))
                return
            except OSError:
                time.sleep(0.2)
    raise RuntimeError(f"Timed out waiting for the local Ground Ryu controller on port {port}.")


def _resolve_ryu_python(root_dir: Path) -> str:
    candidates = [
        root_dir.parent / ".venv39" / "bin" / "python",
        root_dir.parent / ".venv39" / "bin" / "python3",
        Path(sys.executable),
    ]
    for candidate in candidates:
        candidate = Path(candidate)
        if candidate.exists():
            return str(candidate)
    return str(sys.executable)


@contextmanager
def _ground_ryu_controller_context(root_dir: Path, out_dir: Path) -> Iterator[str]:
    port = _pick_free_local_port()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "ryu_ground_controller.log"
    python_bin = _resolve_ryu_python(root_dir)
    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            [
                python_bin,
                "-m",
                "ryu.cmd.manager",
                str(root_dir / "sdn_controller" / "ryu_ground_repeater_app.py"),
                "--wsapi-host",
                "127.0.0.1",
                "--wsapi-port",
                str(port),
            ],
            cwd=str(root_dir),
            env=dict(os.environ),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        try:
            try:
                _wait_for_local_port(port)
            except Exception as exc:
                exit_code = process.poll()
                log_tail = ""
                try:
                    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    if lines:
                        log_tail = "\n".join(lines[-20:])
                except OSError:
                    pass
                detail = f"Local Ground Ryu controller failed to start on port {port}."
                if exit_code is not None:
                    detail += f" Exit code: {exit_code}."
                if log_tail:
                    detail += f"\nLast log lines:\n{log_tail}"
                raise RuntimeError(detail) from exc
            yield f"http://127.0.0.1:{port}"
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def _post_json(url: str, payload: dict, timeout_s: float = 20.0) -> Any:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        try:
            msg = exc.read().decode("utf-8")
        except Exception:
            msg = str(exc)
        raise RuntimeError(f"HTTP {exc.code} on {url}: {msg}") from exc
    except Exception as exc:
        raise RuntimeError(f"POST failed for {url}: {exc}") from exc


class GroundRyuControllerClient:
    def __init__(self, base_url: str, *, max_modes_per_label: int, max_bsm_per_cycle: int) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.max_modes_per_label = int(max_modes_per_label)
        self.max_bsm_per_cycle = int(max_bsm_per_cycle)

    def reset(self) -> None:
        _post_json(
            f"{self.base_url}/sdn/event",
            {
                "type": "RESET",
                "max_modes_per_label": int(self.max_modes_per_label),
                "max_bsm_per_cycle": int(self.max_bsm_per_cycle),
            },
        )

    def submit_request(
        self,
        *,
        req_id: int,
        src: str,
        dst: str,
        f_min: float,
        timeout_ps: int,
        t_ps: int,
        target_pairs: int = 0,
    ) -> None:
        _post_json(
            f"{self.base_url}/sdn/event",
            {
                "type": "ENT_REQUEST",
                "req_id": int(req_id),
                "src": str(src),
                "dst": str(dst),
                "f_min": float(f_min),
                "timeout_ps": int(timeout_ps),
                "t_ps": int(t_ps),
                "target_pairs": int(target_pairs),
            },
        )

    def poll(self, *, req_id: int, t_ps: int) -> list[dict[str, Any]]:
        response = _post_json(
            f"{self.base_url}/sdn/event",
            {
                "type": "POLL",
                "req_id": int(req_id),
                "node": "R1",
                "t_ps": int(t_ps),
            },
        )
        return response if isinstance(response, list) else []

    def update_link_state(self, payload: dict[str, Any]) -> None:
        _post_json(f"{self.base_url}/sdn/fso/update", payload)

    def report_storage(self, *, req_id: int, label: str, accepted_count: int) -> None:
        _post_json(
            f"{self.base_url}/sdn/event",
            {
                "type": "REPORT_STORAGE",
                "req_id": int(req_id),
                "label": str(label),
                "accepted_count": int(accepted_count),
            },
        )

    def report_swap_result(self, *, req_id: int, success_count: int) -> None:
        _post_json(
            f"{self.base_url}/sdn/event",
            {
                "type": "SWAP_RESULT",
                "req_id": int(req_id),
                "success_count": int(success_count),
            },
        )


class GroundLocalControllerClient:
    def __init__(self, *, max_modes_per_label: int, max_bsm_per_cycle: int) -> None:
        self.core = GroundRepeaterSdnControllerCore(
            max_modes_per_label=int(max_modes_per_label),
            max_bsm_per_cycle=int(max_bsm_per_cycle),
        )

    def reset(self) -> None:
        self.core.reset()

    def submit_request(
        self,
        *,
        req_id: int,
        src: str,
        dst: str,
        f_min: float,
        timeout_ps: int,
        t_ps: int,
        target_pairs: int = 0,
    ) -> None:
        self.core.submit_request(
            req_id=int(req_id),
            src=str(src),
            dst=str(dst),
            f_min=float(f_min),
            timeout_ps=int(timeout_ps),
            t_ps=int(t_ps),
            target_pairs=int(target_pairs),
        )

    def poll(self, *, req_id: int, t_ps: int) -> list[dict[str, Any]]:
        del req_id
        return self.core.poll(node="R1", t_ps=int(t_ps))

    def update_link_state(self, payload: dict[str, Any]) -> None:
        self.core.update_link_state(payload)

    def report_storage(self, *, req_id: int, label: str, accepted_count: int) -> None:
        self.core.report_storage(
            req_id=int(req_id),
            label=str(label),
            accepted_count=int(accepted_count),
        )

    def report_swap_result(self, *, req_id: int, success_count: int) -> None:
        self.core.report_swap_result(
            req_id=int(req_id),
            success_count=int(success_count),
        )


def _collect_series(sim) -> dict[str, list[float]]:
    return {
        "t_s": [float(row.t_s) for row in sim.metrics],
        "throughput_pairs_per_s": [float(row.throughput_pairs_per_s) for row in sim.metrics],
        "mean_fidelity_success": [float(row.mean_fidelity_success) for row in sim.metrics],
        "memory_total": [float(row.memory_sat + row.memory_fiber) for row in sim.metrics],
        "sat_visible": [float(row.sat_visible) for row in sim.metrics],
        "sat_elevation_deg": [float(row.sat_elevation_deg) for row in sim.metrics],
    }


def _plot_ground_sweep_family(*, out_dir: Path, filename: str, title: str, series_by_label: Dict[str, dict[str, list[float]]]) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for label, series in series_by_label.items():
        axes[0].plot(series["t_s"], series["throughput_pairs_per_s"], label=label)
        axes[1].plot(series["t_s"], series["mean_fidelity_success"], label=label)
    axes[0].set_ylabel("pairs / s")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("fidelity")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / filename
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _first_visible_window_from_series(series: dict[str, list[float]]) -> tuple[float, float] | None:
    ts = series.get("t_s", [])
    visible = series.get("sat_visible", [])
    if not ts or not visible or len(ts) != len(visible):
        return None
    start_idx = None
    end_idx = None
    in_window = False
    for idx, flag in enumerate(visible):
        if bool(flag) and not in_window:
            start_idx = idx
            in_window = True
        elif not bool(flag) and in_window:
            end_idx = idx - 1
            break
    if in_window and end_idx is None:
        end_idx = len(ts) - 1
    if start_idx is None or end_idx is None:
        return None
    start_s = max(0.0, float(ts[start_idx]) - 20.0)
    end_s = float(ts[end_idx]) + 20.0
    return start_s, end_s


def _plot_ground_paper_rate_fidelity(
    *,
    out_dir: Path,
    t2_series_by_label: Dict[str, dict[str, list[float]]],
    rytov_series_by_label: Dict[str, dict[str, list[float]]],
    xlim_s: tuple[float, float] | None = None,
    suffix: str = "",
) -> tuple[Path, Path]:
    fig_rate, axes_rate = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    fig_fid, axes_fid = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for label, series in t2_series_by_label.items():
        axes_rate[0].plot(series["t_s"], series["throughput_pairs_per_s"], label=label)
        axes_fid[0].plot(series["t_s"], series["mean_fidelity_success"], label=label)
    for label, series in rytov_series_by_label.items():
        axes_rate[1].plot(series["t_s"], series["throughput_pairs_per_s"], label=label)
        axes_fid[1].plot(series["t_s"], series["mean_fidelity_success"], label=label)

    axes_rate[0].set_title("(a) T2 overlays")
    axes_rate[1].set_title("(b) Rytov overlays")
    axes_fid[0].set_title("(a) T2 overlays")
    axes_fid[1].set_title("(b) Rytov overlays")

    for ax in axes_rate:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mean end-to-end rate (pairs / s)")
        ax.grid(True, alpha=0.3)
        if xlim_s is not None:
            ax.set_xlim(*xlim_s)
    for ax in axes_fid:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mean end-to-end fidelity")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        if xlim_s is not None:
            ax.set_xlim(*xlim_s)

    if t2_series_by_label:
        axes_rate[0].legend()
        axes_fid[0].legend()
    if rytov_series_by_label:
        axes_rate[1].legend()
        axes_fid[1].legend()

    fig_rate.suptitle("Ground Repeater DES Mean End-to-End Rate Sweeps\nWindhof OGS to Kirchberg ground node via fiber")
    fig_fid.suptitle("Ground Repeater DES Mean End-to-End Fidelity Sweeps\nWindhof OGS to Kirchberg ground node via fiber")
    fig_rate.tight_layout()
    fig_fid.tight_layout()

    rate_path = out_dir / f"ground_rate_t2_and_rytov_sweeps{suffix}.png"
    fidelity_path = out_dir / f"ground_fidelity_t2_and_rytov_sweeps{suffix}.png"
    fig_rate.savefig(rate_path)
    fig_fid.savefig(fidelity_path)
    plt.close(fig_rate)
    plt.close(fig_fid)
    return rate_path, fidelity_path


def _plot_ground_buffer_t2_family(
    *,
    out_dir: Path,
    sims_by_label: Dict[str, object],
    max_modes_per_label: int,
    xlim_s: tuple[float, float] | None = None,
    suffix: str = "",
) -> Path:
    labels = list(sims_by_label.keys())
    if not labels:
        raise ValueError("need at least one T2 run for the buffer family")
    ncols = 2
    nrows = math.ceil(len(labels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, max(5, 4.4 * nrows)), sharey=True)
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax, label in zip(axes_list, labels):
        sim = sims_by_label[label]
        ts = [row.t_s for row in sim.metrics]
        visible = [row.sat_visible for row in sim.metrics]
        ax.fill_between(
            ts,
            0,
            1,
            where=visible,
            color="#c7d2fe",
            alpha=0.28,
            transform=ax.get_xaxis_transform(),
            label="SAT-OGS visible",
        )
        ax.plot(ts, [row.memory_sat for row in sim.metrics], color="#dc2626", label="Satellite-OGS buffer")
        ax.plot(ts, [row.memory_fiber for row in sim.metrics], color="#2563eb", label="OGS-ground fiber buffer")
        ax.plot(ts, [row.memory_sat + row.memory_fiber for row in sim.metrics], color="#4b5563", linestyle=":", label="Total buffer")
        ax.axhline(max_modes_per_label, color="#111827", linestyle=":", linewidth=1.0, label=f"{max_modes_per_label} per-link")
        ax.axhline(2 * max_modes_per_label, color="#6b7280", linestyle="--", linewidth=1.0, label=f"{2 * max_modes_per_label} total")
        ax.set_title(label)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Buffered pairs")
        ax.grid(True, alpha=0.3)
        if xlim_s is not None:
            ax.set_xlim(*xlim_s)
    for ax in axes_list[len(labels):]:
        ax.axis("off")
    handles, legend_labels = axes_list[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper right")
    fig.suptitle("Windhof-Kirchberg Ground Repeater Buffer Occupancy Over a Pass\nFiber-ground end node and satellite end node with shared OGS repeater")
    fig.tight_layout()
    out_path = out_dir / f"ground_buffer_over_pass_t2_sweep{suffix}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _plot_multi_pass_visibility(
    *,
    out_dir: Path,
    args: argparse.Namespace,
    horizon_hours: float,
    sample_dt_s: float,
) -> tuple[Path, Path]:
    ogs, _ground_end = ground_repeater_sites(args)
    sats = default_satellites(args.tle, args.omm_json)
    epoch_unix_s = float(args.epoch_unix_s) if args.epoch_unix_s is not None else representative_epoch_unix_s(sats)
    selection = select_satellite_for_ground_run(
        sats=sats,
        ogs=ogs,
        epoch_unix_s=epoch_unix_s,
        args=args,
    )
    sat = selection.satellite
    if bool(args.auto_start_at_visibility) and not bool(args.force_visible):
        epoch_unix_s += visibility_start_offset(selection=selection, args=args)

    propagator = SkyfieldPropagator(
        satellites=[sat],
        ground_nodes=[ogs],
        epoch_unix_s=epoch_unix_s,
    )
    geometry = GeometryEngine(propagator)
    end_ps = int(max(1.0, float(horizon_hours) * 3600.0) * PS_PER_S)
    step_ps = int(max(1.0, float(sample_dt_s)) * PS_PER_S)
    contact_windows = geometry.contact_windows(
        ogs.name,
        sat.name,
        start_ps=0,
        end_ps=end_ps,
        step_ps=step_ps,
    )
    t_hours = []
    elevations = []
    visible = []
    t_ps = 0
    while t_ps <= end_ps:
        elevation = propagator.elevation_deg(sat.name, ogs.name, t_ps)
        t_hours.append(t_ps / PS_PER_S / 3600.0)
        elevations.append(elevation)
        visible.append(1 if elevation >= float(args.min_elevation_deg) else 0)
        t_ps += step_ps

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(t_hours, elevations, color="#f59e0b")
    axes[0].fill_between(
        t_hours,
        0,
        1,
        where=visible,
        color="#bfdbfe",
        alpha=0.25,
        transform=axes[0].get_xaxis_transform(),
        label="visible above min elevation",
    )
    axes[0].axhline(float(args.min_elevation_deg), color="#1d4ed8", linestyle="--", linewidth=1.2, label="min elevation")
    axes[0].set_ylabel("elevation (deg)")
    axes[0].set_title("Ground scenario multi-pass satellite visibility")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    bars = []
    peaks = []
    for idx, window in enumerate(contact_windows, start=1):
        start_h = float(window.start_ps) / PS_PER_S / 3600.0
        dur_h = float(window.duration_ps) / PS_PER_S / 3600.0
        bars.append((start_h, dur_h))
        peaks.append(
            {
                "window_index": idx,
                "start_s": float(window.start_ps) / PS_PER_S,
                "end_s": float(window.end_ps) / PS_PER_S,
                "duration_s": float(window.duration_ps) / PS_PER_S,
                "peak_elevation_deg": float(window.peak_elevation_deg),
            }
        )
    if bars:
        axes[1].broken_barh(bars, (0.15, 0.7), facecolors="#0f766e", alpha=0.8)
        for idx, bar in enumerate(bars):
            axes[1].text(
                bar[0] + max(bar[1] / 2.0, 0.02),
                0.55,
                f"{idx + 1}",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
                fontweight="bold",
            )
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_yticks([])
    axes[1].set_ylabel("passes")
    axes[1].set_xlabel("Time from scenario start (hours)")
    axes[1].grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    png_path = out_dir / "ground_multi_pass_visibility.png"
    fig.savefig(png_path)
    plt.close(fig)

    json_path = out_dir / "ground_multi_pass_windows.json"
    payload = {
        "selected_satellite_name": sat.name,
        "visibility_horizon_hours": float(horizon_hours),
        "sample_dt_s": float(sample_dt_s),
        "min_elevation_deg": float(args.min_elevation_deg),
        "windows": peaks,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return png_path, json_path


def _plot_multi_pass_performance(*, out_dir: Path, sim) -> Path:
    ts = [row.t_s / 3600.0 for row in sim.metrics]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(ts, [row.throughput_pairs_per_s for row in sim.metrics], color="#1d4ed8")
    axes[0].fill_between(
        ts,
        0,
        1,
        where=[row.sat_visible for row in sim.metrics],
        color="#bfdbfe",
        alpha=0.25,
        transform=axes[0].get_xaxis_transform(),
        label="satellite visible",
    )
    axes[0].set_ylabel("pairs / s")
    axes[0].set_title("Ground scenario multi-pass end-to-end performance")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(ts, [row.mean_fidelity_success for row in sim.metrics], color="#dc2626")
    axes[1].set_xlabel("Time from scenario start (hours)")
    axes[1].set_ylabel("fidelity")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "ground_multi_pass_throughput_fidelity.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _iter_sweep_values(values: Sequence[float] | Sequence[int]) -> Iterable[float]:
    for value in values:
        yield float(value)


def run_ground_scenario(
    *,
    run_name: str,
    output_root: Path,
    controller_backend: str = "des_only",
    duration_s: float,
    metrics_dt_s: float,
    link_update_dt_s: float,
    gen_delay_s: float,
    swap_dt_s: float,
    source_rate_hz: float,
    max_modes_per_label: int,
    max_bsm_per_cycle: int,
    f_min: float,
    min_elevation_deg: float,
    visibility_scan_s: float,
    visibility_scan_step_s: float,
    visibility_prebuffer_s: float,
    pass_selection: str,
    pass_candidate_limit: int,
    pass_prefilter_step_s: float,
    force_visible: bool,
    fso_params: FSOParams,
    fiber_params: FiberParams,
    fiber_length_km: float,
    ogs_name: str = WINDHOF_OGS_DEFAULT["name"],
    ogs_lat_deg: float = WINDHOF_OGS_DEFAULT["lat_deg"],
    ogs_lon_deg: float = WINDHOF_OGS_DEFAULT["lon_deg"],
    ogs_alt_m: float = WINDHOF_OGS_DEFAULT["alt_m"],
    ground_end_name: str = KIRCHBERG_END_DEFAULT["name"],
    ground_end_lat_deg: float = KIRCHBERG_END_DEFAULT["lat_deg"],
    ground_end_lon_deg: float = KIRCHBERG_END_DEFAULT["lon_deg"],
    ground_end_alt_m: float = KIRCHBERG_END_DEFAULT["alt_m"],
    auto_fiber_length_from_sites: bool = True,
    mem_platform: str,
    mem_t2: float,
    mem_d: float,
    mem_t_mem: float,
    mem_n: float,
    mem_m: float,
    mem_eta_mode0: float,
    mem_eta_ti: float,
    bsm_v_hom: float,
    bsm_eta_det: float,
    bsm_sigma_det_s: float,
    bsm_delta_t_s: float,
    bsm_p_dc: float,
    bsm_swap_model: str,
    seed: int,
    enable_t2_sweep: bool = False,
    t2_sweep_values: Sequence[float] = (),
    enable_rytov_sweep: bool = False,
    rytov_sweep_values: Sequence[float] = (),
    enable_buffer_sweep: bool = False,
    buffer_sweep_values: Sequence[int] = (),
    enable_multi_pass_visibility: bool = False,
    multi_pass_horizon_hours: float = 24.0,
    multi_pass_sample_dt_s: float = 60.0,
) -> GroundRunResult:
    require_qnet_des("the Ground scenario GUI")
    args = _configure_ground_args(
        duration_s=duration_s,
        metrics_dt_s=metrics_dt_s,
        link_update_dt_s=link_update_dt_s,
        gen_delay_s=gen_delay_s,
        swap_dt_s=swap_dt_s,
        source_rate_hz=source_rate_hz,
        max_modes_per_label=max_modes_per_label,
        max_bsm_per_cycle=max_bsm_per_cycle,
        f_min=f_min,
        min_elevation_deg=min_elevation_deg,
        visibility_scan_s=visibility_scan_s,
        visibility_scan_step_s=visibility_scan_step_s,
        visibility_prebuffer_s=visibility_prebuffer_s,
        pass_selection=pass_selection,
        pass_candidate_limit=pass_candidate_limit,
        pass_prefilter_step_s=pass_prefilter_step_s,
        force_visible=force_visible,
        fso_params=fso_params,
        fiber_params=fiber_params,
        fiber_length_km=fiber_length_km,
        ogs_name=ogs_name,
        ogs_lat_deg=ogs_lat_deg,
        ogs_lon_deg=ogs_lon_deg,
        ogs_alt_m=ogs_alt_m,
        ground_end_name=ground_end_name,
        ground_end_lat_deg=ground_end_lat_deg,
        ground_end_lon_deg=ground_end_lon_deg,
        ground_end_alt_m=ground_end_alt_m,
        auto_fiber_length_from_sites=auto_fiber_length_from_sites,
        mem_platform=mem_platform,
        mem_t2=mem_t2,
        mem_d=mem_d,
        mem_t_mem=mem_t_mem,
        mem_n=mem_n,
        mem_m=mem_m,
        mem_eta_mode0=mem_eta_mode0,
        mem_eta_ti=mem_eta_ti,
        bsm_v_hom=bsm_v_hom,
        bsm_eta_det=bsm_eta_det,
        bsm_sigma_det_s=bsm_sigma_det_s,
        bsm_delta_t_s=bsm_delta_t_s,
        bsm_p_dc=bsm_p_dc,
        bsm_swap_model=bsm_swap_model,
        seed=seed,
    )
    args.ground_controller_backend = str(controller_backend).lower().strip()

    out_dir = Path(output_root).expanduser() / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out = str(out_dir)
    sweep_duration_s = max(float(duration_s), float(multi_pass_horizon_hours) * 3600.0) if enable_multi_pass_visibility else float(duration_s)

    controller_mode = str(controller_backend).lower().strip()
    root_dir = Path(__file__).resolve().parent

    def execute_once(run_args: argparse.Namespace):
        return run_once(run_args, controller_client=controller_client)

    if controller_mode == "sdn_ryu":
        controller_context = _ground_ryu_controller_context(root_dir, out_dir)
    else:
        controller_context = nullcontext(None)

    with controller_context as controller_url:
        if controller_mode == "sdn_ryu" and controller_url:
            controller_client = GroundRyuControllerClient(
                controller_url,
                max_modes_per_label=int(args.max_modes_per_label),
                max_bsm_per_cycle=int(args.max_bsm_per_cycle),
            )
        elif controller_mode == "sdn_local":
            controller_client = GroundLocalControllerClient(
                max_modes_per_label=int(args.max_modes_per_label),
                max_bsm_per_cycle=int(args.max_bsm_per_cycle),
            )
        else:
            controller_client = None
        sim = execute_once(args)
        write_outputs(sim, out_dir)

        summary_path = out_dir / "summary.txt"
        summary_path.write_text(_build_summary_text(sim=sim, args=args), encoding="utf-8")

        artifacts = {
            "ground_link_budget_png": out_dir / "ground_link_budget.png",
            "ground_throughput_fidelity_png": out_dir / "ground_throughput_fidelity.png",
            "ground_memory_buffers_png": out_dir / "ground_memory_buffers.png",
        }
        sweep_summary: dict[str, dict[str, dict[str, float]]] = {}

        if enable_t2_sweep and t2_sweep_values:
            t2_series = {}
            t2_sims = {}
            sweep_summary["t2_sweep"] = {}
            for t2_value in _iter_sweep_values(t2_sweep_values):
                sweep_args = argparse.Namespace(**vars(args))
                sweep_args.duration_s = sweep_duration_s
                sweep_args.mem_t2 = t2_value
                sweep_args.no_plots = True
                sweep_args.no_event_log = True
                sim_t2 = execute_once(sweep_args)
                label = f"T2 = {t2_value:g} s"
                t2_series[label] = _collect_series(sim_t2)
                t2_sims[label] = sim_t2
                sweep_summary["t2_sweep"][label] = {
                    "cumulative_successes": float(sim_t2.cumulative_successes),
                    "peak_rate_pairs_per_s": max(t2_series[label]["throughput_pairs_per_s"], default=0.0),
                }
            artifacts["ground_t2_sweep_png"] = _plot_ground_sweep_family(
                out_dir=out_dir,
                filename="ground_t2_sweep.png",
                title="Ground scenario T2 sweep",
                series_by_label=t2_series,
            )
            artifacts["ground_buffer_over_pass_t2_sweep_png"] = _plot_ground_buffer_t2_family(
                out_dir=out_dir,
                sims_by_label=t2_sims,
                max_modes_per_label=int(args.max_modes_per_label),
            )
            single_pass_xlim = _first_visible_window_from_series(next(iter(t2_series.values()))) if t2_series else None
            if single_pass_xlim is not None:
                artifacts["ground_buffer_over_pass_t2_sweep_single_pass_png"] = _plot_ground_buffer_t2_family(
                    out_dir=out_dir,
                    sims_by_label=t2_sims,
                    max_modes_per_label=int(args.max_modes_per_label),
                    xlim_s=single_pass_xlim,
                    suffix="_single_pass",
                )

        if enable_rytov_sweep and rytov_sweep_values:
            rytov_series = {}
            sweep_summary["rytov_sweep"] = {}
            for sigma_value in _iter_sweep_values(rytov_sweep_values):
                sweep_args = argparse.Namespace(**vars(args))
                sweep_args.duration_s = sweep_duration_s
                sweep_args.fso_sigma_r2 = sigma_value
                sweep_args.no_plots = True
                sweep_args.no_event_log = True
                sim_sigma = execute_once(sweep_args)
                label = f"sigma_R^2 = {sigma_value:g}"
                rytov_series[label] = _collect_series(sim_sigma)
                sweep_summary["rytov_sweep"][label] = {
                    "cumulative_successes": float(sim_sigma.cumulative_successes),
                    "peak_rate_pairs_per_s": max(rytov_series[label]["throughput_pairs_per_s"], default=0.0),
                }
            artifacts["ground_rytov_sweep_png"] = _plot_ground_sweep_family(
                out_dir=out_dir,
                filename="ground_rytov_sweep.png",
                title="Ground scenario Rytov sweep",
                series_by_label=rytov_series,
            )

            if enable_t2_sweep and t2_sweep_values:
                paper_rate_path, paper_fidelity_path = _plot_ground_paper_rate_fidelity(
                    out_dir=out_dir,
                    t2_series_by_label=t2_series,
                    rytov_series_by_label=rytov_series,
                )
                artifacts["ground_rate_t2_and_rytov_sweeps_png"] = paper_rate_path
                artifacts["ground_fidelity_t2_and_rytov_sweeps_png"] = paper_fidelity_path
                if single_pass_xlim is not None:
                    zoom_rate_path, zoom_fidelity_path = _plot_ground_paper_rate_fidelity(
                        out_dir=out_dir,
                        t2_series_by_label=t2_series,
                        rytov_series_by_label=rytov_series,
                        xlim_s=single_pass_xlim,
                        suffix="_single_pass",
                    )
                    artifacts["ground_rate_t2_and_rytov_sweeps_single_pass_png"] = zoom_rate_path
                    artifacts["ground_fidelity_t2_and_rytov_sweeps_single_pass_png"] = zoom_fidelity_path

        if enable_buffer_sweep and buffer_sweep_values:
            buffer_series = {}
            sweep_summary["buffer_sweep"] = {}
            for buffer_value in buffer_sweep_values:
                capped_buffer = max(1, int(buffer_value))
                sweep_args = argparse.Namespace(**vars(args))
                sweep_args.duration_s = sweep_duration_s
                sweep_args.max_modes_per_label = capped_buffer
                sweep_args.max_bsm_per_cycle = min(int(args.max_bsm_per_cycle), capped_buffer)
                sweep_args.no_plots = True
                sweep_args.no_event_log = True
                sim_buffer = execute_once(sweep_args)
                label = f"buffer = {capped_buffer}"
                buffer_series[label] = _collect_series(sim_buffer)
                sweep_summary["buffer_sweep"][label] = {
                    "cumulative_successes": float(sim_buffer.cumulative_successes),
                    "peak_rate_pairs_per_s": max(buffer_series[label]["throughput_pairs_per_s"], default=0.0),
                    "peak_total_buffer": max(buffer_series[label]["memory_total"], default=0.0),
                }
            artifacts["ground_buffer_sweep_png"] = _plot_ground_sweep_family(
                out_dir=out_dir,
                filename="ground_buffer_sweep.png",
                title="Ground scenario buffer-capacity sweep",
                series_by_label=buffer_series,
            )

        if enable_multi_pass_visibility:
            multi_args = argparse.Namespace(**vars(args))
            multi_args.duration_s = sweep_duration_s
            multi_args.no_plots = True
            multi_args.no_event_log = True
            sim_multi = execute_once(multi_args)
            multi_pass_png, multi_pass_json = _plot_multi_pass_visibility(
                out_dir=out_dir,
                args=args,
                horizon_hours=multi_pass_horizon_hours,
                sample_dt_s=multi_pass_sample_dt_s,
            )
            artifacts["ground_multi_pass_visibility_png"] = multi_pass_png
            artifacts["ground_multi_pass_windows_json"] = multi_pass_json
            artifacts["ground_multi_pass_throughput_fidelity_png"] = _plot_multi_pass_performance(out_dir=out_dir, sim=sim_multi)

        if sweep_summary:
            sweep_summary_path = out_dir / "ground_sweep_summary.json"
            sweep_summary_path.write_text(json.dumps(sweep_summary, indent=2), encoding="utf-8")
            artifacts["ground_sweep_summary_json"] = sweep_summary_path

        result = GroundRunResult(
            out_dir=out_dir,
            artifacts=artifacts,
            summary_path=summary_path,
            metrics_path=out_dir / "ground_repeater_metrics.csv",
            selected_pass_path=out_dir / "selected_ground_pass.json",
        )
        save_gallery_manifest(result)
        return result
