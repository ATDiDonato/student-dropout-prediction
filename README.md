# Student Dropout Prediction

## Supported saved-model workflow

Use the tracked datasets, saved Stage 1 models/evidence, and the Stage 2/3 rebuilt
model pairs referenced by their `current.json` pointers. Keep their manifests,
metadata, feature schemas, splits and NN scalers together. Incomplete local runs
are not part of the supported release.

The working environment is **Python 3.12.15** on Linux under WSL. Use Python 3.12.
[requirements.txt](requirements.txt) pins the directly used notebook/runtime
packages to their actual installed versions, including IPython and ipykernel.
The core ML pins also match the Stage 2/3 artifact metadata: TensorFlow 2.21.0,
Keras 3.15.1, XGBoost 3.4.1, NumPy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1 and
joblib 1.6.0. These are recorded working versions, not a newly tested clean
installation or a complete transitive dependency lock.

## Environment and kernel

From the cloned repository root, on Linux/WSL:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name student-dropout --display-name "student-dropout"
```

In VS Code, install the Python and Jupyter extensions, select the new `.venv`
interpreter and the `student-dropout` kernel, then start a fresh kernel. The
notebook kernel display name does not install or select an environment. An
existing environment with the recorded versions can also be registered using
that ipykernel command. No machine-specific interpreter path is required.

CPU execution is supported; CUDA is optional for saved-model inference. To force
CPU use, set `CUDA_VISIBLE_DEVICES=-1` before starting the Python process/kernel.

## Run Stage 1 → Stage 2 → Stage 3 in one namespace

Stages 2 and 3 depend on earlier stages' raw and encoded datasets, accumulated
results and trial tables. Separate notebook kernels do not share these variables.
Use the verified VS Code workflow:

1. Open `notebooks/01_stage_1_data_and_modelling.ipynb`, select the configured
   `student-dropout` kernel, start a fresh kernel and choose **Run All**.
2. Keep that same live Stage 1 notebook session. Confirm its working directory
   is the repository's `notebooks/` directory using `%pwd`. If it is the repository
   root, run `%cd notebooks` once in a temporary cell.
3. Add a temporary launcher cell beneath Stage 1 and run:

   ```ipython
   %run -i 02_stage_2_data_and_modelling.ipynb
   ```

4. After Stage 2 completes, add another temporary cell in that same notebook and
   run:

   ```ipython
   %run -i 03_stage_3_data_and_modelling.ipynb
   ```

The `-i` keeps earlier stage variables available to later stages. Stage 2 and
Stage 3 outputs appear beneath their launcher cells in Stage 1; `%run` does not
populate saved outputs in the Stage 2/3 source notebooks. Do not commit the
temporary launcher cells or their outputs. Remove them before saving Stage 1
for a commit, preserving its intended analysis and saved outputs. Restart the
kernel before repeating the complete sequence.

Terminal IPython is an alternative for execution, but equivalent inline plot
display is not guaranteed. From the repository root with the environment active:

```bash
cd notebooks
python -m IPython
```

Run these commands in that same terminal session, in order:

```ipython
%run -i 01_stage_1_data_and_modelling.ipynb
%run -i 02_stage_2_data_and_modelling.ipynb
%run -i 03_stage_3_data_and_modelling.ipynb
```

Leave the committed saved-model defaults unchanged:

- `LOAD_SAVED_TUNING = True`.
- `RUN_XGB_TUNING = RUN_NN_TUNING = False` in all three stages.
- Stage 1: `RUN_XGB_FITTING`, `RUN_NN_FITTING`, `RUN_XGB_STABILITY`,
  `RUN_NN_STABILITY` and `RUN_NN_SURROGATE_FITTING` are `False`.
- Stage 2: `RUN_XGB_REBUILD`, `RUN_NN_REBUILD`, `RUN_NN_STABILITY` and
  `RUN_NN_SURROGATE_FITTING` are `False`.
- Stage 3: `RUN_XGB_REBUILD = RUN_NN_REBUILD = False`.
- `WRITE_DATA_CACHE = False` and `USE_GOOGLE_DRIVE_ARTIFACTS = False` throughout.

Saved summaries provide configuration evidence; they do not imply resumable
search state. Missing or incompatible artifacts should be investigated rather
than worked around by enabling training. Keep data, models and tuning evidence
at their tracked repository-relative locations. Plotting may still write figures
under `reports/`; saved-model mode does not guarantee zero filesystem writes.

## Optional CUDA on Linux/WSL

The Stage 3 NN fixed pair was rebuilt on an NVIDIA GeForce RTX 4050 Laptop GPU,
with the matrix probe and actual training kernels verified on `GPU:0`. For that
optional environment, use an NVIDIA driver exposing CUDA to Linux/WSL and install:

```bash
python -m pip install -r requirements-cuda.txt
```

[requirements-cuda.txt](requirements-cuda.txt) adds TensorFlow's `and-cuda` extra
and pins only its required NVIDIA packages to the observed working versions.
Driver installation is external to pip and is not supplied by this repository.
Other GPUs, drivers and operating systems have not been verified.

The working WSL setup made NVIDIA package libraries discoverable before
launching Python or VS Code. From the activated environment in Bash:

```bash
export LD_LIBRARY_PATH="$(python -c 'import site; from pathlib import Path; print(":".join(sorted({str(p) for root in site.getsitepackages() for p in (Path(root)/"nvidia").glob("*/lib") if p.is_dir()})))')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Launch the session from that shell. With VS Code, start a new WSL window and a
fresh kernel; an existing process may retain its previous environment. Verify
GPU access in a fresh Python process or kernel:

```python
import tensorflow as tf

print(tf.config.list_physical_devices("GPU"))
with tf.device("/GPU:0"):
    result = tf.matmul(tf.ones((2, 2)), tf.ones((2, 2)))
print(result.device)
```

For GPU use, require a listed GPU and a result device containing `GPU:0`.
Availability alone does not prove training placement. If unavailable, the saved
workflow can run on CPU; do not enable rebuild/search to fix setup.

## Verification scope and release limits

Stage 2/3 fixed-configuration XGBoost and NN rebuilds, saved-model readback and
reuse with fitting blocked were verified in the working environment. Their
manifests record successful pairs; metadata distinguishes newly measured metrics
from historical parameter/trial documentation. The shared XGBoost compatibility
path also has mocked-loader regression tests for different checkout locations
and rejection of substantive evidence/data/schema changes. These checks do not
establish a fresh-clone end-to-end notebook run.

Optuna/Keras Tuner searches, stability retraining and other optional fitting
branches are outside this verification scope. Keep their controls disabled.
The separate `demo_model_and_artifact_persistence.ipynb` is outside the supported
workflow for now: its saved history/search-state requirements are incomplete in
the tracked release. Do not include it in the Stage 1 → 2 → 3 sequence.

Colab is not a verified installation target for these pins. If exploring it,
clone under `/content/student-dropout-prediction`, install into that runtime,
and run each notebook's setup cell. Set `COLAB_PROJECT_REPO` before setup if the
clone directory has another name. Keep Drive mode disabled: it redirects reads
away from committed artifacts, and Stage 3 evidence paths currently assume the
repository artifact layout. Enabling Drive is not a supported saved-model setup,
even if files are copied there.

Remaining setup assumptions: availability of these recorded package versions
for the target platform/index, compatible transitive dependencies, and local
Jupyter/kernel configuration have not been tested in a clean environment.
