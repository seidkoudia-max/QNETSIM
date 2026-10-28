# QNETSIM

QNETSIM is a standalone simulator for satellite and hybrid quantum networks.
It includes physical-layer models, mobility, routing, controller interfaces,
plotting, and a QNETSIM-native discrete-event simulation (DES) framework.
An external QCompute installation is not required.

## Applications

| Application | Launcher |
| --- | --- | 
| S1: Single satellite repeater | `bash launch_single_repeater.sh` |
| S2: STBD constellation | `bash launch_stbd.sh` |
| S3: Ground repeater | `bash launch_ground.sh` |
| QUASAR: Matera-Helmos downlinks | `bash launch_quasar.sh`| 
| S3: Physical-timing study | `bash launch_ground_physical.sh`| 

Run launchers from the repository root in an activated environment. Use a
separate terminal for each application. Historical `paper_*` script filenames
are retained for compatibility; they do not change the GUI product names.

## Installation

Clone into a persistent working directory, rather than a temporary folder:

```bash
git clone https://github.com/seidkoudia-max/QNETSIM.git
cd QNETSIM
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` installs DES and local-controller dependencies. It does
**not** install Ryu. S1, S2 and the legacy S3 execution profile offer
`DES only`, `DES + SDN Local` and `DES + SDN Ryu`. The S3 physical-timing
profile and QUASAR studies are DES-only workflows.

### Optional Ryu Environment

The existing Ryu 4.34 integration uses a legacy Python 3.9 environment.
Python 3.9 and the pinned Eventlet dependency are legacy components: use
this environment for local research, not as an internet-facing service.
Ryu installation on Python 3.12 is not supported by this setup.

If you need Ryu, create a separate environment and launch the GUI from it:

```bash
python3.9 -m venv .venv-ryu39
source .venv-ryu39/bin/activate
python -m pip install --upgrade pip
python -m pip install 'setuptools<67' wheel
python -m pip install --no-build-isolation -r requirements-ryu.txt
bash launch_stbd.sh
```



## Updated Studies

The Ground GUI offers a physical-timing execution profile for the
Windhof-Kirchberg scenario. This separates optical loading, causal heralding,
finite memories, BSM service and endpoint delivery. It reports raw rates,
fidelity-qualified rates and Bell-vector conditional fidelity. A compact
simulated pass fixture is included under `qnetsim/data/s3_windhof_kirchberg/`.

The QUASAR GUI covers 810 nm and 1550 nm dual downlinks, background and
coincidence modelling, detector presets, polarization compensation and
coincidence-window/source-rate studies. Parsed scenario parameters are
included.


## Outputs and Tests

Applications create run directories under `qnetsim/Output/` by default.


From the repository root:

```bash
python -m unittest discover -s tests -p 'test_*.py'
cd qnetsim
python -m unittest discover -s quasar -p 'test_*.py'
python -m unittest discover -s studies/manuscript_reproduction -p 'test_*.py'
python -m unittest discover -s studies/quasar_matera_helmos -p 'test_*.py'
```


