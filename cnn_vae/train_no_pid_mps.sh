#!/usr/bin/env bash
set -euo pipefail

# Run this from the repository root:
#   bash cnn_vae/train_no_pid_mps.sh

CACHE="cnn_vae/cache_jet_images_32_no_pid"
OUTDIR="cnn_vae/runs/no_pid_beta_1_$(date +%Y%m%d_%H%M%S)"

EPOCHS=600
BATCH_SIZE=256
LR=1e-3
LATENT_DIM=32
HIDDEN_DIMS="32,64,128,256,512"
LOSS_TYPE="H"
RECON_LOSS_TYPE="mse"
RECON_LOSS_WEIGHT=1.0
RECON_LOG_X0=1e-3
BETA=1.0
GAMMA=1000.0
MAX_CAPACITY=25
CAP_MAX_ITER=100000
PATIENCE=20
N_GENERATE=64

mkdir -p "$OUTDIR"

python cnn_vae/code_training/train_betavae_mps.py \
    --cache-dir "$CACHE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --latent-dim "$LATENT_DIM" \
    --hidden-dims "$HIDDEN_DIMS" \
    --loss-type "$LOSS_TYPE" \
    --recon-loss-type "$RECON_LOSS_TYPE" \
    --recon-loss-weight "$RECON_LOSS_WEIGHT" \
    --recon-log-x0 "$RECON_LOG_X0" \
    --beta "$BETA" \
    --gamma "$GAMMA" \
    --max-capacity "$MAX_CAPACITY" \
    --cap-max-iter "$CAP_MAX_ITER" \
    --patience "$PATIENCE" \
    --n-generate "$N_GENERATE" \
    --output "$OUTDIR/best_model_betavae.pth" \
    2>&1 | tee "$OUTDIR/train.log"

echo "Training log: $OUTDIR/train.log"
echo "Loss plot:    $OUTDIR/best_model_betavae_training_curves.png"
