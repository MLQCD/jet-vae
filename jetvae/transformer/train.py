"""Train the shape-only (z, eta, phi) Particle Transformer Sinkhorn-EMD VAE.

    python -m jetvae.transformer.train --config configs/nopid_beta1e-5.yaml \
        --cache-dir cache/ --output runs/s1/best_model.pth --seed 1
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ..data.dataset import CachedParticleShapeDataset, collate_cached_batch
from .model import ParticleTransformerShapeEMDVAE


def select_device(preference: str) -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if preference == "cuda":
        raise RuntimeError("CUDA was requested but is not available")
    if (
        preference in ("auto", "mps")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    if preference == "mps":
        raise RuntimeError("MPS was requested but is not available")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def annealed_sinkhorn_epsilon(
    epoch: int, start: float, final: float | None, anneal_epochs: int
) -> float:
    if final is None or anneal_epochs <= 0:
        return start
    progress = min(1.0, epoch / float(anneal_epochs))
    return float(start * (final / start) ** progress)


def annealed_sinkhorn_n_iters(
    epoch: int, start: int, final: int | None, anneal_epochs: int
) -> int:
    if final is None or anneal_epochs <= 0:
        return start
    progress = min(1.0, epoch / float(anneal_epochs))
    return int(round(start * (final / start) ** progress))


def atomic_json_dump(payload, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(payload, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def build_datasets(args, parser):
    cache_dir = Path(args.cache_dir)
    meta_path = cache_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    args.max_particles = int(meta.get("max_particles", args.max_particles))
    train_dataset = CachedParticleShapeDataset(
        cache_dir / "train_particles.npy", cache_dir / "train_mask.npy")
    val_dataset = CachedParticleShapeDataset(
        cache_dir / "val_particles.npy", cache_dir / "val_mask.npy")
    expected_shape = (args.max_particles, 3)
    for name, ds in (("train", train_dataset), ("val", val_dataset)):
        if ds.particles.shape[1:] != expected_shape:
            parser.error(f"Cache {name} shape is {ds.particles.shape[1:]}, "
                         f"but metadata/config expects {expected_shape}")
    return train_dataset, val_dataset, collate_cached_batch


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    train,
    grad_clip,
    kl_weight_mode,
    kl_scale,
    stochastic_validation,
):
    model.train(train)
    totals = {key: 0.0 for key in ("loss", "emd", "kld", "weighted_kld")}
    examples = 0
    for batch_index, batch in enumerate(loader, start=1):
        particles = batch["particles"].to(device, non_blocking=True)
        target = batch["momenta"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)

        kld_weight = (
            particles.shape[0] / len(loader.dataset)
            if kl_weight_mode == "original"
            else 1.0
        )
        with torch.set_grad_enabled(train):
            sample_latent = train or stochastic_validation
            losses = model.loss_function(
                *model(particles, mask, target, sample_latent=sample_latent),
                M_N=kld_weight,
                kl_scale=kl_scale,
            )
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError(f"Non-finite loss in batch {batch_index}")
            if train:
                losses["loss"].backward()
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        batch_size = particles.shape[0]
        examples += batch_size
        values = {
            "loss": losses["loss"],
            "emd": losses["EMD_Loss"],
            "kld": losses["KLD"],
            "weighted_kld": losses["Weighted_KLD"],
        }
        for key, value in values.items():
            totals[key] += float(value.detach()) * batch_size
    return {key: value / examples for key, value in totals.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Particle Transformer VAE with shape-only (z, eta, phi) Sinkhorn EMD loss"
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "YAML file of --flag-name (dashes or underscores) -> value, applied as "
            "new defaults before CLI flags are parsed. Explicit CLI flags still "
            "override the config file, which still overrides this parser's own "
            "built-in defaults."
        ),
    )
    parser.add_argument("--cache-dir", required=False, default=None,
                        help="Directory written by jetvae.data.preprocess (required).")
    parser.add_argument("--max-particles", type=int, default=80)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=1e-5)
    parser.add_argument(
        "--kl-warmup-epochs",
        type=int,
        default=0,
        help="Linearly ramp the KL multiplier from 0 to 1 over this many epochs.",
    )
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.05)
    parser.add_argument(
        "--sinkhorn-epsilon-final",
        type=float,
        default=0.01,
        help=(
            "If set, geometrically anneal sinkhorn_epsilon from "
            "--sinkhorn-epsilon down to this value over "
            "--sinkhorn-epsilon-anneal-epochs, then hold. Validation always "
            "uses this final value (once set)."
        ),
    )
    parser.add_argument(
        "--sinkhorn-epsilon-anneal-epochs",
        type=int,
        default=50,
        help=(
            "Number of epochs over which to anneal to --sinkhorn-epsilon-final "
            "(also used as the schedule length for --sinkhorn-n-iters-final)."
        ),
    )
    parser.add_argument(
        "--sinkhorn-n-iters-final",
        type=int,
        default=150,
        help="If set, geometrically anneal sinkhorn_n_iters up to this value.",
    )
    parser.add_argument("--sinkhorn-n-iters", type=int, default=50)
    parser.add_argument("--sinkhorn-p", type=float, default=1.0)
    parser.add_argument(
        "--angular-scale",
        type=float,
        default=0.4,
        help="Jet clustering radius R; the ground cost is deltaR_ij / R.",
    )
    parser.add_argument("--no-sinkhorn-debias", action="store_true")
    parser.add_argument("--kl-weight-mode", choices=("one", "original"), default="one")
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-generate", type=int, default=64)
    parser.add_argument(
        "--stochastic-validation",
        action="store_true",
        help="Sample z during validation; by default validation decodes posterior mu.",
    )
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument(
        "--resume",
        default=None,
        help="Resume model, optimizer, scheduler, history, and early stopping state.",
    )
    initialization.add_argument(
        "--init-checkpoint",
        default=None,
        help="Initialize model weights only.",
    )
    parser.add_argument(
        "--output",
        default="runs/default/best_model.pth",
    )
    return parser


def checkpoint_payload(
    *,
    config,
    epoch,
    best_val,
    stale_epochs,
    history,
    model,
    optimizer,
    scheduler,
    train_generator,
):
    return {
        **config,
        "epoch": int(epoch),
        "val_loss": float(best_val),
        "stale_epochs": int(stale_epochs),
        "history": history,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_generator_state": train_generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def restore_rng_state(checkpoint) -> None:
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])
    cuda_states = checkpoint.get("cuda_rng_state_all")
    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])


def load_config_defaults(config_path: str, parser: argparse.ArgumentParser) -> dict:
    """Load a YAML file of flag -> value and validate its keys against `parser`.

    Keys may use dashes or underscores (``sinkhorn-epsilon`` or
    ``sinkhorn_epsilon``); both map to the same argparse destination.
    """
    with open(config_path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        parser.error(f"--config {config_path} must contain a YAML mapping")
    defaults = {str(key).replace("-", "_"): value for key, value in raw.items()}
    valid_dests = set(vars(parser.parse_args([])).keys())
    unknown = sorted(set(defaults) - valid_dests)
    if unknown:
        parser.error(f"Unknown key(s) in --config {config_path}: {unknown}")
    return defaults


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    pre_args, _ = parser.parse_known_args(argv)
    if pre_args.config:
        parser.set_defaults(**load_config_defaults(pre_args.config, parser))
    args = parser.parse_args(argv)
    if not args.cache_dir:
        parser.error("--cache-dir is required (build it with python -m jetvae.data.preprocess)")
    positive_integer_args = {
        "epochs": args.epochs,
        "batch-size": args.batch_size,
        "sinkhorn-n-iters": args.sinkhorn_n_iters,
        "patience": args.patience,
        "n-generate": args.n_generate,
    }
    for name, value in positive_integer_args.items():
        if value <= 0:
            parser.error(f"--{name} must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.kl_warmup_epochs < 0:
        parser.error("--kl-warmup-epochs cannot be negative")
    if args.sinkhorn_epsilon_anneal_epochs < 0:
        parser.error("--sinkhorn-epsilon-anneal-epochs cannot be negative")
    if args.sinkhorn_epsilon_final is not None:
        if args.sinkhorn_epsilon_final <= 0.0:
            parser.error("--sinkhorn-epsilon-final must be positive")
        if args.sinkhorn_epsilon_anneal_epochs <= 0:
            parser.error(
                "--sinkhorn-epsilon-anneal-epochs must be positive when "
                "--sinkhorn-epsilon-final is set"
            )
    if args.sinkhorn_n_iters_final is not None:
        if args.sinkhorn_n_iters_final <= 0:
            parser.error("--sinkhorn-n-iters-final must be positive")
        if args.sinkhorn_n_iters_final < args.sinkhorn_n_iters:
            parser.error(
                "--sinkhorn-n-iters-final should be >= --sinkhorn-n-iters"
            )
        if args.sinkhorn_epsilon_anneal_epochs <= 0:
            parser.error(
                "--sinkhorn-epsilon-anneal-epochs must be positive when "
                "--sinkhorn-n-iters-final is set"
            )
    if not 0.0 < args.val_fraction < 1.0:
        parser.error("--val-fraction must lie strictly between 0 and 1")
    if args.lr <= 0.0:
        parser.error("--lr must be positive")

    seed_everything(args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    config_path = output.with_name(f"{output.stem}_config.json")
    history_path = output.with_name(f"{output.stem}_history.json")
    last_checkpoint_path = output.with_name(f"{output.stem}_last.pth")

    train_dataset, val_dataset, collate_fn = build_datasets(args, parser)
    device = select_device(args.device)
    train_generator = torch.Generator().manual_seed(args.seed)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    model = ParticleTransformerShapeEMDVAE(
        max_particles=args.max_particles,
        latent_dim=args.latent_dim,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        dropout=args.dropout,
        beta=args.beta,
        reconstruction_weight=args.reconstruction_weight,
        sinkhorn_epsilon=args.sinkhorn_epsilon,
        sinkhorn_n_iters=args.sinkhorn_n_iters,
        sinkhorn_p=args.sinkhorn_p,
        angular_scale=args.angular_scale,
        sinkhorn_debias=not args.no_sinkhorn_debias,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    config = vars(args).copy()
    config.update(model_type="particle_transformer_shape_sinkhorn_emd_vae")
    atomic_json_dump(config, config_path)

    history: list[dict[str, float | int]] = []
    best_val = float("inf")
    stale_epochs = 0
    start_epoch = 1

    if args.init_checkpoint:
        initialization = torch.load(
            args.init_checkpoint, map_location=device, weights_only=False
        )
        state_dict = dict(initialization.get("model_state_dict", initialization))
        incompatibility = model.load_state_dict(state_dict, strict=False)
        if incompatibility.missing_keys or incompatibility.unexpected_keys:
            raise RuntimeError(
                "Initialization checkpoint is not architecture-compatible: "
                f"missing={incompatibility.missing_keys}, "
                f"unexpected={incompatibility.unexpected_keys}"
            )
        print(
            f"Initialized model weights from {args.init_checkpoint}; optimizer starts fresh."
        )

    if args.resume:
        resumed = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resumed["model_state_dict"])
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        if "scheduler_state_dict" in resumed:
            scheduler.load_state_dict(resumed["scheduler_state_dict"])
        start_epoch = int(resumed["epoch"]) + 1
        best_val = float(resumed.get("val_loss", float("inf")))
        stale_epochs = int(resumed.get("stale_epochs", 0))
        history = list(resumed.get("history", []))
        restore_rng_state(resumed)
        if "train_generator_state" in resumed:
            train_generator.set_state(resumed["train_generator_state"].cpu())
        print(
            f"Resumed {args.resume} at epoch {start_epoch}; "
            f"best validation loss={best_val:.6g}."
        )

    print("=" * 72)
    print("Particle Transformer Shape Sinkhorn-EMD VAE")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Device: {device}; train={len(train_dataset)}; val={len(val_dataset)}")
    print(
        f"angular_scale(R)={args.angular_scale}; epsilon={args.sinkhorn_epsilon}; "
        f"iterations={args.sinkhorn_n_iters}; debias={not args.no_sinkhorn_debias}"
    )
    print(
        f"beta={args.beta:g}; KL warmup={args.kl_warmup_epochs} epochs; "
        f"validation={'stochastic' if args.stochastic_validation else 'posterior mean'}"
    )
    print(f"Best checkpoint: {output}; recovery checkpoint: {last_checkpoint_path}")
    print("=" * 72, flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        kl_scale = (
            1.0
            if args.kl_warmup_epochs == 0
            else min(1.0, epoch / float(args.kl_warmup_epochs))
        )
        train_epsilon = annealed_sinkhorn_epsilon(
            epoch, args.sinkhorn_epsilon, args.sinkhorn_epsilon_final,
            args.sinkhorn_epsilon_anneal_epochs,
        )
        train_n_iters = annealed_sinkhorn_n_iters(
            epoch, args.sinkhorn_n_iters, args.sinkhorn_n_iters_final,
            args.sinkhorn_epsilon_anneal_epochs,
        )
        model.sinkhorn_epsilon = train_epsilon
        model.sinkhorn_n_iters = train_n_iters
        train_metrics = run_epoch(
            model, train_loader, optimizer, device, True, args.grad_clip,
            args.kl_weight_mode, kl_scale, args.stochastic_validation,
        )
        model.sinkhorn_epsilon = (
            args.sinkhorn_epsilon_final
            if args.sinkhorn_epsilon_final is not None
            else args.sinkhorn_epsilon
        )
        model.sinkhorn_n_iters = (
            args.sinkhorn_n_iters_final
            if args.sinkhorn_n_iters_final is not None
            else args.sinkhorn_n_iters
        )
        val_metrics = run_epoch(
            model, val_loader, optimizer, device, False, args.grad_clip,
            args.kl_weight_mode, 1.0, args.stochastic_validation,
        )
        scheduler.step(val_metrics["loss"])
        history.append(
            {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_kl_scale": kl_scale,
                "train_sinkhorn_epsilon": train_epsilon,
                "train_sinkhorn_n_iters": train_n_iters,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
        )
        print(
            f"Epoch {epoch:03d} | eps={train_epsilon:.5f} n_iters={train_n_iters:4d} | "
            f"train loss={train_metrics['loss']:.6f} emd={train_metrics['emd']:.6f} "
            f"kld={train_metrics['kld']:.6f} | "
            f"val loss={val_metrics['loss']:.6f} emd={val_metrics['emd']:.6f} "
            f"kld={val_metrics['kld']:.6f} "
            f"train_kl_scale={kl_scale:.3f}",
            flush=True,
        )

        is_best = val_metrics["loss"] < best_val
        if is_best:
            best_val = val_metrics["loss"]
            stale_epochs = 0
        else:
            stale_epochs += 1

        payload = checkpoint_payload(
            config=config, epoch=epoch, best_val=best_val, stale_epochs=stale_epochs,
            history=history, model=model, optimizer=optimizer, scheduler=scheduler,
            train_generator=train_generator,
        )
        atomic_torch_save(payload, last_checkpoint_path)
        atomic_json_dump(history, history_path)
        if is_best:
            atomic_torch_save(payload, output)
            print(f"  -> saved best checkpoint {output}", flush=True)
        if stale_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if not output.exists():
        if not last_checkpoint_path.exists():
            raise RuntimeError(
                "No checkpoint was produced. If resuming, set --epochs above the "
                "checkpoint epoch or keep the original best checkpoint available."
            )
        atomic_torch_save(
            torch.load(last_checkpoint_path, map_location="cpu", weights_only=False),
            output,
        )

    checkpoint = torch.load(output, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        z_frac, eta, phi = model.sample(args.n_generate, device)
    generated = torch.stack((z_frac, eta, phi), dim=-1).cpu().numpy()
    np.savez(
        output.with_name(f"{output.stem}_generated_particles.npz"),
        shape=generated,
        columns=np.asarray(["z", "eta", "phi"]),
    )
    print(f"Saved generated particles next to {output}")


main_with_args = main


if __name__ == "__main__":
    main()
