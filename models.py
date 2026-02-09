
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class ModelBundle:
    name: str
    model: Any
    framework: str  # "sklearn" or "torch"

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.framework == "sklearn":
            return self.model.predict(X)
        else:
            self.model.eval()
            with torch.no_grad():
                logits = self.model(torch.from_numpy(X).float())
                return torch.argmax(logits, dim=1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.framework == "sklearn":
            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(X)
            preds = self.model.predict(X)
            n_classes = int(preds.max()) + 1
            proba = np.zeros((len(preds), n_classes), dtype=float)
            proba[np.arange(len(preds)), preds] = 1.0
            return proba

        # torch model: softmax over logits
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X).float())
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs


class SimpleMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def build_model(cfg: Dict, n_features: int, n_classes: int, seed: int) -> ModelBundle:
    mtype = cfg["model"]["type"].lower().strip()

    if mtype == "logreg":
        params = cfg["model"].get("logreg", {})
        model = LogisticRegression(
            max_iter=int(params.get("max_iter", 500)),
            solver="lbfgs",
        )
        return ModelBundle(name="LogisticRegression", model=model, framework="sklearn")

    if mtype == "rf":
        params = cfg["model"].get("rf", {})
        model = RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=params.get("max_depth", None),
            random_state=seed,
            n_jobs=-1,
        )
        return ModelBundle(name="RandomForest", model=model, framework="sklearn")

    if mtype == "mlp":
        params = cfg["model"].get("mlp", {})
        hidden_dim = int(params.get("hidden_dim", 64))
        model = SimpleMLP(n_features, hidden_dim, n_classes)
        return ModelBundle(name="SimpleMLP", model=model, framework="torch")

    raise ValueError(f"Unknown model.type: {mtype}. Use logreg | rf | mlp.")


def train_model(bundle: ModelBundle, cfg: Dict, data) -> Dict[str, Any]:
    """
    Trains the model and returns training metadata (epochs, loss).
    For sklearn models, fitting is immediate.
    For torch MLP, we implement a minimal training loop.
    """
    if bundle.framework == "sklearn":
        bundle.model.fit(data.X_train, data.y_train)
        return {"trainer": "sklearn.fit"}

    # Torch training loop (small + reproducible)
    params = cfg["model"].get("mlp", {})
    epochs = int(params.get("epochs", 10))
    lr = float(params.get("lr", 1e-3))
    batch_size = int(params.get("batch_size", 128))

    torch.manual_seed(int(cfg["run"]["seed"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle.model.to(device)

    ds = TensorDataset(
        torch.from_numpy(data.X_train).float(),
        torch.from_numpy(data.y_train).long(),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(bundle.model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    last_loss = None
    bundle.model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = bundle.model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            last_loss = float(loss.item())

    return {"trainer": "torch", "device": device, "epochs": epochs, "lr": lr, "last_batch_loss": last_loss}
