# QNETSIM Standalone Bundle

This folder is a GitHub-ready standalone bundle of the active QNETSIM apps:

- `QNETSIM Single Repeater`
- `QNETSIM STBD`
- `QNETSIM Ground`

It includes the repository physics, mobility, plotting, controller, and optimizer
modules plus a QNETSIM-native DES/framework layer so the bundle runs without a
separate checkout of the larger QCompute/QNET tree.

## Folder layout

- `qnetsim/`
  - Streamlit GUIs, simulators, physics, mobility, algorithms, controllers
- `qnetsim/framework/`
  - local DES core and protocol compatibility layer used by the simulator
- `requirements.txt`
  - Python dependencies for the standalone bundle
- `launch_single_repeater.sh`
- `launch_stbd.sh`
- `launch_ground.sh`

## Recommended Python version

Use Python `3.9` if possible, since the current simulator environment has been
validated most heavily there.

## Quick start

```bash
cd standalone_bundle
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Launch the Single Repeater GUI

```bash
cd qnetsim
python -m streamlit run paper_figure_studio.py --server.port 8501
```

Open [http://localhost:8501](http://localhost:8501)

### Launch the STBD GUI

```bash
cd qnetsim
python -m streamlit run paper_stbd_studio.py --server.port 8502
```

Open [http://localhost:8502](http://localhost:8502)

### Launch the Ground GUI

```bash
cd qnetsim
python -m streamlit run paper_ground_studio.py --server.port 8503
```

Open [http://localhost:8503](http://localhost:8503)

## Optional Ryu-backed controller modes

The bundle includes the local Ryu controller apps used by:

- `DES + Ryu` in the Single Repeater GUI
- `SDN + Ryu` in the STBD GUI
- `SDN + Ryu` in the Ground GUI

Those modes use the same Python environment after `ryu` is installed from
`requirements.txt`.

## Outputs

Generated files are written under:

- `qnetsim/Output/`

The apps create run-specific subfolders automatically.

## Notes on the standalone framework layer

The simulator uses a QNETSIM-native DES/framework layer that preserves the
runtime interfaces required by the three published GUIs:

- `qnetsim/framework/core`
- `qnetsim/framework/protocols`

This lets the standalone repository run independently of external QNET/Baidu
DES sources while keeping the current app behavior intact.

## Publish this bundle to GitHub

This standalone bundle can be published as its own repository.

```bash
cd standalone_bundle
git init -b main
git add .
git commit -m "Initial QNETSIM standalone bundle"
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

If you prefer GitHub CLI, replace the remote-creation steps with your normal
`gh repo create ...` workflow after authenticating with GitHub on your machine.
