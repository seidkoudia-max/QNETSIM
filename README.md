# QNETSIM 

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
  - Python dependencies for DES-only and DES + SDN Local
- `requirements-ryu.txt`
  - Optional add-on dependencies for DES + SDN Ryu
- `launch_single_repeater.sh`
- `launch_stbd.sh`
- `launch_ground.sh`

## Recommended Python version

Use Python `3.9`. This is the recommended baseline for the standalone bundle
because it supports all three controller modes, including `DES + SDN Ryu`.

The optional `ryu` package does not currently build reliably on Python `3.12`,
so if you want the full simulator feature set, create the virtual environment
with Python `3.9`.

## Quick start

```bash
cd standalone_bundle
/opt/homebrew/bin/python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-ryu.txt
```

This is the recommended full setup and gives you:

- `DES only`
- `DES + SDN Local`
- `DES + SDN Ryu`

If `python3.9` lives somewhere else on your machine, use that executable
instead of `/opt/homebrew/bin/python3.9`.

### Minimal non-Ryu fallback

If you only want `DES only` and `DES + SDN Local`, you can still use a newer
Python such as `3.12` and install:

```bash
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

The bundle includes the local Ryu controller apps used by:

- `DES + SDN Ryu` in the Single Repeater GUI
- `DES + SDN Ryu` in the STBD GUI
- `DES + SDN Ryu` in the Ground GUI

Those modes use the same Python environment after `ryu` is installed from
`requirements-ryu.txt`.

## Outputs

Generated files are written under:

- `qnetsim/Output/`

The apps create run-specific subfolders automatically.

## Notes on the standalone framework layer

The simulator uses a QNETSIM-native DES/framework layer that preserves the
runtime interfaces required by the three published GUIs:

- `qnetsim/framework/core`
- `qnetsim/framework/protocols`

This lets the standalone repository run independently of external sources while keeping the current app behavior intact.


