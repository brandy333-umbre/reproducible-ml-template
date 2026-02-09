
from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import yaml
import joblib

from io_utils import now_utc_compact, ensure_dir, save_json, copy_file
from data import make_synthetic_classification
from models import build_model, train_model
from eval import evaluate_model


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Config-driven, reproducible ML training template.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    run_name = cfg["run"]["name"]
    seed = int(cfg["run"]["seed"])
    output_root = cfg["run"]["output_dir"]

    run_id = f"{run_name}_{now_utc_compact()}"
    run_dir = os.path.join(output_root, run_id)
    ensure_dir(run_dir)

    # Snapshot config for reproducibility
    copy_file(args.config, os.path.join(run_dir, "config_snapshot.yaml"))

    # 1) Data (deterministic)
    dcfg = cfg["data"]
    data = make_synthetic_classification(
        seed=seed,
        n_train=int(dcfg["n_train"]),
        n_val=int(dcfg["n_val"]),
        n_test=int(dcfg["n_test"]),
        n_features=int(dcfg["n_features"]),
        n_classes=int(dcfg["n_classes"]),
        class_sep=float(dcfg["class_sep"]),
    )

    # 2) Model
    model_bundle = build_model(cfg, n_features=int(dcfg["n_features"]), n_classes=int(dcfg["n_classes"]), seed=seed)

    # 3) Train
    train_meta = train_model(model_bundle, cfg, data)

    # 4) Eval
    eval_out = evaluate_model(model_bundle, data)

    # 5) Save artifacts
    # Save model (joblib works for sklearn; for torch, save state_dict too)
    if model_bundle.framework == "sklearn":
        joblib.dump(model_bundle.model, os.path.join(run_dir, "model.joblib"))
        model_artifact = {"type": model_bundle.name, "framework": "sklearn", "path": "model.joblib"}
    else:
        import torch
        torch.save(model_bundle.model.state_dict(), os.path.join(run_dir, "model_state.pt"))
        model_artifact = {"type": model_bundle.name, "framework": "torch", "path": "model_state.pt"}

    # Save metrics + metadata
    save_json(os.path.join(run_dir, "metrics.json"), eval_out)
    save_json(os.path.join(run_dir, "run_meta.json"), {
        "run_id": run_id,
        "run_name": run_name,
        "seed": seed,
        "model": model_artifact,
        "train_meta": train_meta,
    })

    # Print summary
    print(f"\nRun saved to: {run_dir}")
    print(f"Model: {model_bundle.name} ({model_bundle.framework})")
    print(f"VAL : acc={eval_out['val']['accuracy']:.3f} f1={eval_out['val']['macro_f1']:.3f} logloss={eval_out['val']['logloss']:.3f}")
    print(f"TEST: acc={eval_out['test']['accuracy']:.3f} f1={eval_out['test']['macro_f1']:.3f} logloss={eval_out['test']['logloss']:.3f}\n")


if __name__ == "__main__":
    main()
