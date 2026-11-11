#!/usr/bin/env python3
"""Visualize Beta-VAE training logs and latent active dimensions."""

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from betaVae import BetaVAE


EPOCH_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)\s+\|\s+"
    r"train_loss=(?P<train_loss>[0-9.]+)\s+"
    r"\(recon=(?P<train_recon>[0-9.]+)\s+kld=(?P<train_kld>[0-9.]+)\)\s+\|\s+"
    r"val_loss=(?P<val_loss>[0-9.]+)\s+"
    r"\(recon=(?P<val_recon>[0-9.]+)\s+kld=(?P<val_kld>[0-9.]+)\)"
)


def parse_epoch_rows(log_path: Path) -> list[dict[str, float]]:
    rows = []
    for line in log_path.read_text().splitlines():
        match = EPOCH_RE.search(line)
        if match is None:
            continue
        row = {"epoch": int(match.group("epoch"))}
        for key in (
            "train_loss",
            "train_recon",
            "train_kld",
            "val_loss",
            "val_recon",
            "val_kld",
        ):
            row[key] = float(match.group(key))
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No epoch summaries found in {log_path}")
    return rows


def load_best_model(checkpoint_path: Path) -> tuple[BetaVAE, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = BetaVAE(
        in_channels=int(checkpoint["n_channels"]),
        latent_dim=int(checkpoint["latent_dim"]),
        image_size=int(checkpoint["img_size"]),
        hidden_dims=list(checkpoint["hidden_dims"]),
        loss_type="H",
        recon_loss_type=checkpoint.get("recon_loss_type", "mse"),
        recon_log_x0=float(checkpoint.get("recon_log_x0", 1e-3)),
        output_mode=checkpoint.get("output_mode", "event_pt"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def compute_latent_activity(
    model: BetaVAE,
    cache_dir: Path,
    split: str,
    batch_size: int,
    threshold: float,
) -> dict:
    images = np.load(cache_dir / f"{split}_images.npy", mmap_mode="r")
    dataset = TensorDataset(torch.from_numpy(np.asarray(images, dtype=np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    mus = []
    log_vars = []
    for (x,) in loader:
        mu, log_var = model.encode(x.float())
        mus.append(mu.cpu())
        log_vars.append(log_var.cpu())

    mu = torch.cat(mus, dim=0)
    log_var = torch.cat(log_vars, dim=0)
    mu_var = mu.var(dim=0, unbiased=False)
    mean_kl = (-0.5 * (1 + log_var - mu.pow(2) - log_var.exp())).mean(dim=0)
    active_mask = mu_var > threshold

    return {
        "mu_variance": mu_var.numpy(),
        "mean_kl_per_dim": mean_kl.numpy(),
        "active_mask": active_mask.numpy(),
        "active_dimensions": int(active_mask.sum().item()),
        "n_samples": int(mu.shape[0]),
    }


def write_epoch_csv(rows: list[dict[str, float]], output_path: Path) -> None:
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_latent_csv(activity: dict, output_path: Path) -> None:
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dim", "mu_variance", "mean_kl", "active_var_gt_threshold"],
        )
        writer.writeheader()
        for dim, (mu_var, mean_kl, active) in enumerate(
            zip(
                activity["mu_variance"],
                activity["mean_kl_per_dim"],
                activity["active_mask"],
            )
        ):
            writer.writerow(
                {
                    "dim": dim,
                    "mu_variance": float(mu_var),
                    "mean_kl": float(mean_kl),
                    "active_var_gt_threshold": int(active),
                }
            )


def save_plot(
    rows: list[dict[str, float]],
    activity: dict,
    checkpoint: dict,
    threshold: float,
    output_path: Path,
) -> None:
    best_row = min(rows, key=lambda row: row["val_loss"])
    epochs = [row["epoch"] for row in rows]
    latent_dim = int(checkpoint["latent_dim"])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05])

    panels = [
        ("loss", "Total Loss", "loss"),
        ("recon", "Reconstruction Loss", "loss"),
        ("kld", "KLD Loss", "KLD"),
    ]
    for axis, (key, title, ylabel) in zip(
        [fig.add_subplot(grid[0, i]) for i in range(3)],
        panels,
    ):
        axis.plot(
            epochs,
            [row[f"train_{key}"] for row in rows],
            label="train",
            color="#1f77b4",
            linewidth=1.8,
        )
        axis.plot(
            epochs,
            [row[f"val_{key}"] for row in rows],
            label="val",
            color="#d62728",
            linewidth=1.8,
        )
        axis.axvline(
            best_row["epoch"],
            color="#2ca02c",
            linestyle="--",
            linewidth=1.2,
            label=f"best epoch {best_row['epoch']}" if key == "loss" else None,
        )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.legend()

    axis = fig.add_subplot(grid[1, :])
    dims = np.arange(latent_dim)
    mu_var = activity["mu_variance"]
    mean_kl = activity["mean_kl_per_dim"]
    active_mask = activity["active_mask"]
    colors = np.where(active_mask, "#2ca02c", "#9aa0a6")

    axis.bar(dims, mu_var, color=colors, label="Var_x(mu_j)")
    axis.axhline(
        threshold,
        color="#d62728",
        linestyle="--",
        linewidth=1.3,
        label=f"active threshold {threshold:g}",
    )
    axis2 = axis.twinx()
    axis2.plot(
        dims,
        mean_kl,
        color="#1f77b4",
        marker="o",
        linewidth=1.3,
        markersize=3,
        label="mean KL per dim",
    )
    axis.set_title(
        "Best model latent activity: "
        f"{activity['active_dimensions']}/{latent_dim} active dimensions"
    )
    axis.set_xlabel("Latent dimension index")
    axis.set_ylabel("Variance of encoder mean")
    axis2.set_ylabel("Mean KL per dimension")
    axis.set_xticks(dims)
    axis.set_yscale("log")
    positive_mu_var = mu_var[mu_var > 0]
    y_min = max(float(positive_mu_var.min()) * 0.5, 1e-8)
    y_max = max(float(mu_var.max()) * 2.0, threshold * 2.0)
    axis.set_ylim(bottom=y_min, top=y_max)
    handles1, labels1 = axis.get_legend_handles_labels()
    handles2, labels2 = axis2.get_legend_handles_labels()
    axis.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    fig.suptitle(
        "Beta-VAE training from train.log | "
        f"best val loss {best_row['val_loss']:.5f} at epoch {best_row['epoch']} | "
        f"checkpoint epoch {checkpoint.get('epoch')}",
        fontsize=13,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Beta-VAE train.log losses and best-checkpoint latent activity."
    )
    parser.add_argument("--log", required=True, type=Path, help="Path to train.log")
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to best_model_betavae.pth",
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
        type=Path,
        help="Cache directory containing val_images.npy",
    )
    parser.add_argument("--split", default="val", help="Cache split to encode")
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument(
        "--active-threshold",
        default=1e-2,
        type=float,
        help="Active unit threshold for Var_x(mu_j)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG path. Defaults to training_losses_and_latent_activity.png next to log.",
    )
    parser.add_argument(
        "--epoch-csv",
        type=Path,
        help="Optional parsed epoch CSV path.",
    )
    parser.add_argument(
        "--latent-csv",
        type=Path,
        help="Optional per-latent-dimension CSV path.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Optional summary JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.log.parent / "training_losses_and_latent_activity.png"
    epoch_csv = args.epoch_csv or args.log.parent / "parsed_epoch_losses.csv"
    latent_csv = args.latent_csv or args.log.parent / "latent_activity_best_model.csv"
    summary_json = args.summary_json or args.log.parent / "latent_activity_summary.json"

    rows = parse_epoch_rows(args.log)
    model, checkpoint = load_best_model(args.checkpoint)
    activity = compute_latent_activity(
        model,
        args.cache_dir,
        args.split,
        args.batch_size,
        args.active_threshold,
    )

    write_epoch_csv(rows, epoch_csv)
    write_latent_csv(activity, latent_csv)
    save_plot(rows, activity, checkpoint, args.active_threshold, output)

    best_row = min(rows, key=lambda row: row["val_loss"])
    summary = {
        "log_path": str(args.log),
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_val_loss": float(checkpoint.get("val_loss", float("nan"))),
        "best_epoch_in_log": int(best_row["epoch"]),
        "best_val_loss_in_log": float(best_row["val_loss"]),
        "latent_dim": int(checkpoint["latent_dim"]),
        "active_dimension_definition": (
            f"Var_x(mu_j) > {args.active_threshold:g} on {args.split} images"
        ),
        "active_threshold": float(args.active_threshold),
        "active_dimensions": int(activity["active_dimensions"]),
        "validation_samples": int(activity["n_samples"]),
        "mu_variance": activity["mu_variance"].tolist(),
        "mean_kl_per_dim": activity["mean_kl_per_dim"].tolist(),
        "plot_path": str(output),
        "epoch_csv": str(epoch_csv),
        "latent_csv": str(latent_csv),
    }
    summary_json.write_text(json.dumps(summary, indent=2))

    print(f"Saved plot: {output}")
    print(f"Saved parsed epochs: {epoch_csv}")
    print(f"Saved latent activity: {latent_csv}")
    print(f"Saved summary: {summary_json}")
    print(
        "Active latent dimensions: "
        f"{activity['active_dimensions']}/{int(checkpoint['latent_dim'])}"
    )


if __name__ == "__main__":
    main()
