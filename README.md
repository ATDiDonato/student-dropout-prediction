# Student Dropout Prediction

## Running locally

Run the notebooks from the cloned repository so repo-relative paths resolve as expected.

1. Create and activate your environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Open the notebooks from [notebooks](/home/alextd/projects/student-dropout-prediction/notebooks) in VS Code / WSL and run them normally.

The notebooks and shared path helpers resolve the project root from the repo, so saved artefacts continue to land in the standard project folders:

- `data/`
- `models/`
- `tuning/`
- `reports/`
- `demo_artifacts/`

## Running in Google Colab

Colab runtimes are temporary, so clone the repo and reinstall dependencies at the start of each new session.

1. Clone the repo into `/content`:

```bash
!git clone <repo-url> /content/student-dropout-prediction
```

2. Move into the repo and install dependencies:

```bash
%cd /content/student-dropout-prediction
!pip install -r requirements.txt
```

3. Open a notebook from the cloned repo, ideally under [notebooks](/home/alextd/projects/student-dropout-prediction/notebooks), and run the Colab setup cell near the top.

The setup cell will:

- detect Colab
- switch into the cloned repo notebook directory when needed
- add the repo root to `sys.path`
- keep the same repo-relative save/load behaviour used locally

The default Colab assumption is `/content/student-dropout-prediction`. If you clone into a different folder name, set:

```python
import os
os.environ["COLAB_PROJECT_REPO"] = "<your-cloned-folder-name>"
```

before running the notebook setup cell.

Optional: you can mount Google Drive if you want longer-lived artefacts, but the notebooks do not require Drive and continue to save into the repo by default.
