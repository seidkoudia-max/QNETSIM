"""Non-destructive, allowlisted export from the development workspace.

Run only on an update branch. The native framework, Git history, requirements,
launchers and standalone documentation are preserved. Generated results and
private source documents are deliberately outside the export allowlist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

BUNDLE = Path(__file__).resolve().parents[1]
APP = BUNDLE / "qnetsim"
FILES = [
    "paper_figure_studio.py", "paper_figure_studio_backend.py", "paper_stbd_studio.py",
    "stbd_figure_studio_backend.py", "paper_ground_studio.py", "paper_ground_studio_backend.py",
    "des_no_sdn_paper_figures.py", "des_sdn_paper_figures.py", "des_physics_no_sdn_simulator.py",
    "des_stbd_physics_simulator.py", "des_ground_repeater_simulator.py", "des_ground_physical.py",
    "des_stbd_physical_layer.py", "ground_physical_studio.py", "quasar_gui.py", "quasar_des_simulator.py",
    "starlink_tle.txt", "starlink_multi.json", "starlink_multi.tle",
    "sim/__init__.py", "sim/seeding.py", "sim/paper_network.py",
    "studies/quasar_matera_helmos/run_study.py", "studies/quasar_matera_helmos/inputs.json",
    "studies/quasar_matera_helmos/test_study.py",
    "studies/manuscript_reproduction/s3_physical_timing.py",
    "studies/manuscript_reproduction/test_s3_timing.py",
    "studies/manuscript_reproduction/test_s3_timing_gui.py",
]
TREES = ["algorithms", "channels", "mobility", "quantum", "stbd_constellation", "quasar", "assets"]
CONTROLLERS = ["__init__.py", "link_state_db.py", "path_engine.py", "repeater_controller_core.py",
    "ryu_repeater_controller_app.py", "stbd_segment_controller_core.py", "ryu_stbd_constellation_app.py",
    "ground_repeater_controller_core.py", "ryu_ground_repeater_app.py"]


def portable(text, relative):
    if relative == "des_physics_no_sdn_simulator.py":
        start = text.index("QCOMPUTE_ROOT =")
        end = text.index("from channels.fso import", start)
        text = text[:start] + ("from framework.core.des import DESEnv, Entity, EventHandler\n\n"
            "HAVE_QNET_DES = True\nQNET_DES_IMPORT_ERROR = None\n\n"
            "def require_qnet_des(context: str = 'this workflow') -> None:\n"
            "    # Compatibility name; the standalone executor is QNETSIM-native.\n"
            "    return None\n\n") + text[end:]
    text = text.replace("standalone_bundle.qnetsim.framework", "framework")
    text = text.replace("standalone_bundle/qnetsim/framework/", "framework/")
    text = text.replace("Extensions.QuantumNetwork.qcompute_qnet.core", "framework.core")
    text = text.replace("Extensions.QuantumNetwork.qcompute_qnet.protocols", "framework.protocols")
    if relative == "studies/manuscript_reproduction/s3_physical_timing.py":
        text = text.replace("from studies.manuscript_reproduction.s3_evidence import mean_ci",
                            "from studies.manuscript_reproduction.s3_study_utils import mean_ci")
        text = text.replace("from studies.manuscript_reproduction.stochasticity import read_rows",
                            "from studies.manuscript_reproduction.s3_study_utils import read_rows")
        text = text.replace('ROOT/"Output/reviewer_s3_sampled_20260926"', 'ROOT/"data/s3_windhof_kirchberg"')
        text = text.replace('<p><a href="../reviewer_s3_swap_audit_20260927/REPORT.html">Previous processing audit</a>.</p>',
                            '<p>Historical audit results are not included in the standalone source export.</p>')
    if relative == "quasar/test_des.py":
        text = text.replace('Path("/Users/seid.koudia/Downloads/QUASAR simulation parameters.xlsx")',
                            'Path(__file__).parent/"optional_inputs/QUASAR simulation parameters.xlsx"')
    if relative == "studies/quasar_matera_helmos/test_study.py":
        text = text.replace("import unittest\n", "import unittest\nimport tempfile\nfrom pathlib import Path\nfrom unittest.mock import patch\n")
        text = text.replace('        summary=json.loads((s.OUT/"summary.json").read_text())',
            '        with tempfile.TemporaryDirectory() as directory:\n'
            '            with patch.object(s, "OUT", Path(directory)), patch.object(s, "plots"):\n'
            '                s.main()\n'
            '                summary=json.loads((s.OUT/"summary.json").read_text())')
    if relative.endswith(".md"):
        text = text.replace("cd /Users/seid.koudia/QCompute/qnetsim", "cd QNETSIM/qnetsim")
    if "Extensions.QuantumNetwork.qcompute_qnet" in text:
        raise ValueError(f"Unsupported upstream dependency in {relative}")
    return text


def sync(source):
    if (BUNDLE/".git").exists():
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=BUNDLE, text=True).strip()
        if branch in {"main", "master", ""}:
            raise ValueError("Switch to an update branch before exporting")
    paths = set(FILES + ["sdn_controller/"+x for x in CONTROLLERS])
    for directory in TREES:
        for path in (source/directory).rglob("*"):
            if path.is_file() and path.suffix in {".py", ".json", ".md", ".txt", ".svg", ".png", ".jpg", ".geojson"}:
                paths.add(path.relative_to(source).as_posix())
    records = {}
    for relative in sorted(paths):
        src, dst = source/relative, APP/relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix in {".py", ".md"}:
            dst.write_text(portable(src.read_text(), relative))
        else:
            shutil.copy2(src, dst)
        records[relative] = hashlib.sha256(dst.read_bytes()).hexdigest()
    # Small simulated replay fixture; never export whole Output directories.
    for relative in ("t2_600/arguments.json", "t2_600/ground_optical_trace.csv", "figure3_pass_geometry.csv"):
        src = source/"Output/reviewer_s3_sampled_20260926"/relative
        dst = APP/"data/s3_windhof_kirchberg"/relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Normalize line endings while preserving every replay value.
        dst.write_text(src.read_text())
        records[dst.relative_to(APP).as_posix()] = hashlib.sha256(dst.read_bytes()).hexdigest()
    manifest = dict(schema_version=2, app_root="qnetsim", framework="QNETSIM-native",
        apps={"S1":"paper_figure_studio.py", "S2":"paper_stbd_studio.py", "S3":"paper_ground_studio.py",
              "S3 physical timing":"ground_physical_studio.py", "QUASAR":"quasar_gui.py"},
        exported_sha256=records, excluded=["virtual environments", "generated results", "source workbooks",
            "manuscript PDFs", "private local documents"],
        note="Existing framework, repository metadata and release infrastructure are preserved")
    (BUNDLE/"bundle_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"Exported {len(records)} allowlisted files; native framework and Git metadata preserved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    sync(parser.parse_args().source.resolve())
