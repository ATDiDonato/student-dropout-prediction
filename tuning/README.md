# Tuning Artifacts

Stage tuning outputs are stored under `tuning/stage_<n>/` with one folder per model family.

## Stage 1

- `tuning/stage_1/xgboost/` contains the Optuna-backed XGBoost tuning artefacts, including `best_params.json`, `trials.csv`, `optuna_study.sqlite3`, and the exported best model.
- `tuning/stage_1/neural_network/` keeps the canonical refined-search outputs in `best_params.json`, `trials.csv`, and `search_summary.txt`.
- `tuning/stage_1/neural_network/keras_tuner/nn_tabular_auc/` preserves the original broad Keras Tuner search.
- `tuning/stage_1/neural_network/keras_tuner/nn_tabular_auc_refined/` preserves the refined follow-up search used for the saved default parameters.
- `initial_best_params.json`, `initial_trials.csv`, and `initial_search_summary.txt` are lightweight exports derived from the preserved initial search so both Stage 1 tuning runs remain inspectable in a public clone.
