# reproducible-ml-template
a small framework that lets you run ML experiments without editing code each time.


Instead of changing code every run, you edit a single `config.yaml` file to control:
- dataset generation
- model type + hyperparameters
- run name, seed, and output directory

Each run produces a clean artifact bundle:
- trained model
- metrics JSON
- run metadata (including seed and training details)
- config snapshot (so the run can be reproduced later)
