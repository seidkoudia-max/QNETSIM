from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Dict, Iterable, Iterator, List

from channels.fso import FSOAnalyticalChannelModel, FSOParams
from des_no_sdn_paper_figures import (
    BSMStudyConfig,
    CITY_PAIRS,
    MEMORY_PLATFORM_LABELS,
    MemoryStudyConfig,
    OPTIMIZED_ANNUAL_CITY_PAIRS,
    REPRESENTATIVE_OVERPASSES,
    annualised_des_buffered_curve_optimized_geometry,
    annualised_pdv_curve,
    annualised_pdv_curve_optimized_geometry,
    default_paper_out_dir,
    plot_altitude_dependence,
    plot_buffer_consumption,
    plot_buffer_over_pass_examples,
    plot_buffered_annual_throughput,
    plot_case_overview_composite,
    plot_des_fidelity,
    plot_des_parameter_sweeps,
    plot_optimized_annual_pair_distribution,
    plot_paris_vienna_buffer_over_altitudes,
    plot_paris_vienna_buffer_over_t2,
    plot_representative_losses,
    plot_representative_rates,
    representative_trace,
    require_qnet_des,
    run_des_fidelity_stats,
    study_fso_params,
    write_csv_rows,
)
from des_sdn_paper_figures import generate_sdn_studio_plots


@dataclass(frozen=True)
class StudioPlotSelection:
    representative_overpasses: bool = True
    archetype_sweeps: bool = True
    annual_throughput: bool = True
    buffered_annual_throughput: bool = True
    paris_vienna_buffer_altitudes: bool = True
    paris_vienna_buffer_t2: bool = True
    buffer_examples: bool = False
    buffer_consumption: bool = False
    overview_composite: bool = False


BACKEND_OPTIONS = {
    "des_only": "DES only",
    "sdn_inproc": "DES + SDN Local",
    "sdn_ryu": "DES + SDN Ryu",
}


def default_studio_out_root() -> Path:
    return Path(default_paper_out_dir()).resolve().parent / "gui_studio_runs"


def memory_platform_options() -> List[str]:
    return list(MEMORY_PLATFORM_LABELS.keys())


def backend_options() -> List[str]:
    return list(BACKEND_OPTIONS.keys())


def backend_label(backend_key: str) -> str:
    return BACKEND_OPTIONS.get(backend_key, backend_key)


def _gather_existing_images(out_dir: Path, patterns: Iterable[str]) -> Dict[str, Path]:
    found: Dict[str, Path] = {}
    for pattern in patterns:
        for path in sorted(out_dir.glob(pattern)):
            title = path.stem.replace("_", " ").title()
            found[title] = path
    return found


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
    raise RuntimeError(f"Timed out waiting for the local Ryu controller on port {port}.")


@contextmanager
def _ryu_controller_context(root_dir: Path, out_dir: Path) -> Iterator[str]:
    port = _pick_free_local_port()
    log_path = out_dir / "ryu_controller.log"
    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "ryu.cmd.manager",
                str(root_dir / "sdn_controller" / "ryu_repeater_controller_app.py"),
                "--wsapi-host",
                "127.0.0.1",
                "--wsapi-port",
                str(port),
            ],
            cwd=str(root_dir),
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
                detail = f"Local Ryu controller failed to start on port {port}."
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


def generate_backend_plots(
    out_dir: Path,
    *,
    backend_key: str,
    selection: StudioPlotSelection,
    fso_params: FSOParams,
    memory_config: MemoryStudyConfig,
    bsm_config: BSMStudyConfig,
    source_rate_hz: float,
    des_source_rate_hz: float,
    ssqr_efficiency: float,
    min_elevation_deg: float,
    des_repeats: int,
    sweep_repeats: int,
    seed: int,
    annual_contact_fraction: float,
    t2_values_s: Iterable[float],
    rytov_values: Iterable[float],
    paris_vienna_altitudes_km: Iterable[float],
) -> Dict[str, Path]:
    backend_key = str(backend_key).strip().lower()
    if backend_key == "des_only":
        return generate_studio_plots(
            out_dir,
            selection=selection,
            fso_params=fso_params,
            memory_config=memory_config,
            bsm_config=bsm_config,
            source_rate_hz=source_rate_hz,
            des_source_rate_hz=des_source_rate_hz,
            ssqr_efficiency=ssqr_efficiency,
            min_elevation_deg=min_elevation_deg,
            des_repeats=des_repeats,
            sweep_repeats=sweep_repeats,
            seed=seed,
            annual_contact_fraction=annual_contact_fraction,
            t2_values_s=t2_values_s,
            rytov_values=rytov_values,
            paris_vienna_altitudes_km=paris_vienna_altitudes_km,
        )

    if backend_key not in {"sdn_inproc", "sdn_ryu"}:
        raise ValueError(f"unknown backend key: {backend_key}")

    root_dir = Path(__file__).resolve().parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if backend_key == "sdn_inproc":
        return generate_sdn_studio_plots(
            out_dir,
            selection=selection.__dict__,
            fso_params=fso_params,
            memory_config=memory_config,
            bsm_config=bsm_config,
            source_rate_hz=source_rate_hz,
            des_source_rate_hz=des_source_rate_hz,
            ssqr_efficiency=ssqr_efficiency,
            min_elevation_deg=min_elevation_deg,
            des_repeats=des_repeats,
            sweep_repeats=sweep_repeats,
            seed=seed,
            annual_contact_fraction=annual_contact_fraction,
            t2_values_s=t2_values_s,
            rytov_values=rytov_values,
            paris_vienna_altitudes_km=paris_vienna_altitudes_km,
            controller_backend="inproc",
            controller_url="",
        )

    with _ryu_controller_context(root_dir, out_dir) as controller_url:
        return generate_sdn_studio_plots(
            out_dir,
            selection=selection.__dict__,
            fso_params=fso_params,
            memory_config=memory_config,
            bsm_config=bsm_config,
            source_rate_hz=source_rate_hz,
            des_source_rate_hz=des_source_rate_hz,
            ssqr_efficiency=ssqr_efficiency,
            min_elevation_deg=min_elevation_deg,
            des_repeats=des_repeats,
            sweep_repeats=sweep_repeats,
            seed=seed,
            annual_contact_fraction=annual_contact_fraction,
            t2_values_s=t2_values_s,
            rytov_values=rytov_values,
            paris_vienna_altitudes_km=paris_vienna_altitudes_km,
            controller_backend="ryu",
            controller_url=controller_url,
        )


def generate_studio_plots(
    out_dir: Path,
    *,
    selection: StudioPlotSelection,
    fso_params: FSOParams,
    memory_config: MemoryStudyConfig,
    bsm_config: BSMStudyConfig,
    source_rate_hz: float,
    des_source_rate_hz: float,
    ssqr_efficiency: float,
    min_elevation_deg: float,
    des_repeats: int,
    sweep_repeats: int,
    seed: int,
    annual_contact_fraction: float,
    t2_values_s: Iterable[float],
    rytov_values: Iterable[float],
    paris_vienna_altitudes_km: Iterable[float],
) -> Dict[str, Path]:
    require_qnet_des("the QNETSIM figure studio")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated: Dict[str, Path] = {}
    fso_model = FSOAnalyticalChannelModel(fso_params)
    t2_values = [float(value) for value in t2_values_s]
    rytov_values_list = [float(value) for value in rytov_values]
    altitude_values = [float(value) for value in paris_vienna_altitudes_km]

    traces: Dict[str, List[Dict[str, float]]] = {}
    altitude_rows: Dict[str, List[Dict[str, float]]] = {}
    optimized_rows: Dict[str, List[Dict[str, float]]] = {}
    buffered_rows: Dict[str, List[Dict[str, float]]] = {}

    needs_representative = selection.representative_overpasses or selection.overview_composite
    needs_reference_altitude = selection.overview_composite
    needs_optimized = (
        selection.annual_throughput
        or selection.buffered_annual_throughput
        or selection.paris_vienna_buffer_altitudes
        or selection.paris_vienna_buffer_t2
        or selection.buffer_examples
        or selection.buffer_consumption
    )
    needs_buffered = (
        selection.buffered_annual_throughput
        or selection.paris_vienna_buffer_altitudes
        or selection.paris_vienna_buffer_t2
        or selection.buffer_examples
        or selection.buffer_consumption
    )

    if needs_representative:
        combined_trace_rows: List[Dict[str, float]] = []
        for scenario in REPRESENTATIVE_OVERPASSES:
            rows = representative_trace(
                scenario,
                fso_model=fso_model,
                min_elevation_deg=min_elevation_deg,
                source_rate_hz=source_rate_hz,
                ssqr_efficiency=ssqr_efficiency,
            )
            traces[scenario.key] = rows
            for row in rows:
                combined_trace_rows.append({"scenario": scenario.key, **row})
        write_csv_rows(out_dir / "representative_overpasses.csv", combined_trace_rows)

        plot_representative_losses(
            REPRESENTATIVE_OVERPASSES,
            traces,
            out_dir,
            fso_model=fso_model,
        )
        plot_representative_rates(REPRESENTATIVE_OVERPASSES, traces, out_dir)
        generated["Representative Downlink Loss"] = out_dir / "representative_downlink_loss.png"
        generated["Representative Distribution Rates"] = (
            out_dir / "representative_distribution_rates.png"
        )

        fidelity_rows: List[Dict[str, float]] = []
        stats_by_scenario: Dict[str, Dict[str, List[float]]] = {}
        for index, scenario in enumerate(REPRESENTATIVE_OVERPASSES):
            stats = run_des_fidelity_stats(
                scenario,
                repeats=des_repeats,
                seed=seed + 1000 * index,
                min_elevation_deg=min_elevation_deg,
                source_rate_hz=des_source_rate_hz,
                fso_params=fso_params,
                memory_config=memory_config,
                bsm_config=bsm_config,
            )
            stats_by_scenario[scenario.key] = stats
            for idx, t_s in enumerate(stats["time_s"]):
                fidelity_rows.append(
                    {
                        "scenario": scenario.key,
                        "time_s": t_s,
                        "fidelity_median": stats["fidelity_median"][idx],
                        "fidelity_q1": stats["fidelity_q1"][idx],
                        "fidelity_q3": stats["fidelity_q3"][idx],
                        "rate_mean": stats["rate_mean"][idx],
                        "rate_std": stats["rate_std"][idx],
                    }
                )
        write_csv_rows(out_dir / "des_fidelity_stats.csv", fidelity_rows)
        plot_des_fidelity(REPRESENTATIVE_OVERPASSES, stats_by_scenario, out_dir)
        generated["DES Fidelity and Rate"] = out_dir / "des_fidelity_and_ssqr_rate.png"

    if selection.archetype_sweeps:
        plot_des_parameter_sweeps(
            REPRESENTATIVE_OVERPASSES,
            out_dir,
            min_elevation_deg=min_elevation_deg,
            source_rate_hz=des_source_rate_hz,
            seed=seed + 50000,
            t2_values_s=t2_values,
            rytov_values=rytov_values_list,
            sweep_repeats=sweep_repeats,
            base_fso_params=fso_params,
            memory_config=memory_config,
            bsm_config=bsm_config,
        )
        generated["Archetype Rate Sweeps"] = (
            out_dir / "archetype_des_ssqr_rate_t2_and_rytov_sweeps.png"
        )
        generated["Archetype Fidelity Sweeps"] = (
            out_dir / "archetype_des_fidelity_t2_and_rytov_sweeps.png"
        )

    if needs_reference_altitude:
        altitudes = list(range(200, 1001, 20))
        combined_altitude_rows: List[Dict[str, float]] = []
        for pair in CITY_PAIRS:
            rows = annualised_pdv_curve(
                pair,
                altitudes_km=altitudes,
                fso_model_factory=lambda _min_el: FSOAnalyticalChannelModel(fso_params),
                min_elevation_deg=min_elevation_deg,
                source_rate_hz=source_rate_hz,
                ssqr_efficiency=ssqr_efficiency,
                annual_contact_fraction=annual_contact_fraction,
            )
            altitude_rows[pair.key] = rows
            for row in rows:
                combined_altitude_rows.append(
                    {
                        "pair": pair.key,
                        "pair_distance_km": pair.baseline_km,
                        "track_angle_deg": pair.track_angle_deg,
                        **row,
                    }
                )
        write_csv_rows(out_dir / "altitude_dependence.csv", combined_altitude_rows)
        plot_altitude_dependence(altitude_rows, out_dir)
        generated["Altitudinal PDV Dependence"] = out_dir / "altitudinal_pdv_dependence.png"

    if needs_optimized:
        optimized_combined_rows: List[Dict[str, float]] = []
        for pair in OPTIMIZED_ANNUAL_CITY_PAIRS:
            rows = annualised_pdv_curve_optimized_geometry(
                pair,
                altitudes_km=list(range(200, 1001, 20)),
                fso_model_factory=lambda _min_el: FSOAnalyticalChannelModel(fso_params),
                min_elevation_deg=min_elevation_deg,
                source_rate_hz=source_rate_hz,
                ssqr_efficiency=ssqr_efficiency,
                annual_contact_fraction=annual_contact_fraction,
            )
            optimized_rows[pair.key] = rows
            for row in rows:
                optimized_combined_rows.append(
                    {
                        "pair": pair.key,
                        "pair_distance_km": pair.baseline_km,
                        **row,
                    }
                )
        write_csv_rows(
            out_dir / "optimized_annual_throughput_paris_vienna_warsaw_lisbon.csv",
            optimized_combined_rows,
        )

        if selection.annual_throughput:
            plot_optimized_annual_pair_distribution(
                OPTIMIZED_ANNUAL_CITY_PAIRS,
                optimized_rows,
                out_dir,
            )
            generated["Optimized Annual Throughput"] = (
                out_dir / "optimized_annual_throughput_paris_vienna_warsaw_lisbon.png"
            )

    if needs_buffered:
        buffered_combined_rows: List[Dict[str, float]] = []
        for index, pair in enumerate(OPTIMIZED_ANNUAL_CITY_PAIRS):
            rows = annualised_des_buffered_curve_optimized_geometry(
                pair,
                altitudes_km=list(range(200, 1001, 40)),
                min_elevation_deg=min_elevation_deg,
                source_rate_hz=des_source_rate_hz,
                annual_contact_fraction=annual_contact_fraction,
                seed=seed + 70000 + 1000 * index,
                memory_config=memory_config,
                bsm_config=bsm_config,
            )
            buffered_rows[pair.key] = rows
            for row in rows:
                buffered_combined_rows.append(
                    {
                        "pair": pair.key,
                        "pair_distance_km": pair.baseline_km,
                        **row,
                    }
                )
        write_csv_rows(
            out_dir / "buffered_annual_throughput_paris_vienna_warsaw_lisbon.csv",
            buffered_combined_rows,
        )

        if selection.buffered_annual_throughput:
            plot_buffered_annual_throughput(
                OPTIMIZED_ANNUAL_CITY_PAIRS,
                buffered_rows,
                optimized_rows,
                out_dir,
                memory_config=memory_config,
            )
            generated["Buffered Annual Throughput"] = (
                out_dir / "buffered_annual_throughput_paris_vienna_warsaw_lisbon.png"
            )

        if selection.buffer_consumption:
            plot_buffer_consumption(
                OPTIMIZED_ANNUAL_CITY_PAIRS,
                buffered_rows,
                out_dir,
                max_modes_per_label=memory_config.max_modes_per_label,
            )
            generated["Buffer Consumption"] = (
                out_dir / "buffer_consumption_paris_vienna_warsaw_lisbon.png"
            )

        if selection.buffer_examples:
            plot_buffer_over_pass_examples(
                OPTIMIZED_ANNUAL_CITY_PAIRS,
                buffered_rows,
                out_dir,
                min_elevation_deg=min_elevation_deg,
                source_rate_hz=des_source_rate_hz,
                seed=seed + 90000,
                base_fso_params=fso_params,
                memory_config=memory_config,
                bsm_config=bsm_config,
            )
            generated["Buffer Over Pass Examples"] = out_dir / "buffer_over_pass_examples.png"

        if selection.paris_vienna_buffer_altitudes:
            plot_paris_vienna_buffer_over_altitudes(
                buffered_rows,
                out_dir,
                min_elevation_deg=min_elevation_deg,
                source_rate_hz=des_source_rate_hz,
                seed=seed + 91000,
                base_fso_params=fso_params,
                memory_config=memory_config,
                bsm_config=bsm_config,
                altitudes_km=altitude_values,
            )
            generated["Paris-Vienna Buffer vs Altitude"] = (
                out_dir / "paris_vienna_buffer_over_pass_altitudes.png"
            )

        if selection.paris_vienna_buffer_t2:
            plot_paris_vienna_buffer_over_t2(
                buffered_rows,
                out_dir,
                min_elevation_deg=min_elevation_deg,
                source_rate_hz=des_source_rate_hz,
                seed=seed + 92000,
                base_fso_params=fso_params,
                memory_config=memory_config,
                bsm_config=bsm_config,
                t2_values_s=t2_values,
            )
            generated["Paris-Vienna Buffer vs T2"] = (
                out_dir / "paris_vienna_buffer_over_pass_t2_sweep.png"
            )

    if selection.overview_composite:
        plot_case_overview_composite(out_dir)
        generated["Representative Composite"] = (
            out_dir / "paper_representative_case_overview_composite.png"
        )

    return generated
