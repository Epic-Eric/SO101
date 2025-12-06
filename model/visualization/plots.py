import json
import os
from typing import List

import matplotlib.pyplot as plt


def plot_metrics_json(metrics_json_path: str, out_path: str):
    with open(metrics_json_path, "r") as f:
        rows = json.load(f)
    epochs = [r["epoch"] for r in rows]
    loss = [r["loss"] for r in rows]
    rec = [r["rec_loss"] for r in rows]
    kld = [r["kld"] for r in rows]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, label="loss")
    plt.plot(epochs, rec, label="rec_loss")
    plt.plot(epochs, kld, label="kld")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Metrics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
