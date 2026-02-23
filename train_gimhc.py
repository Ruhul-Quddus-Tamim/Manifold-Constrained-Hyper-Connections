"""Training script for GI-mHC: Gated Injection Hyper-Connections.

Trains a GIHyperTransformer on any HuggingFace character-level LM dataset
using the full corpus (no held-out split). After training, saves:
  - output/loss_and_grad_norm.png  — smoothed loss + gradient norm curves
  - output/gate_values.png         — learned gate strengths per layer
"""

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from mhc.data import (
    BATCH_SIZE,
    LM_SEQ_LEN,
    MAX_SEQ_LEN,
    create_dataloaders,
    create_lm_dataloader_full,
)
from mhc.models_gimhc import GIHyperTransformer, build_gimhc_config
from mhc.train_utils import (
    autoregressive_generate,
    get_deepseek_step_scheduler,
    train,
)


def plot_loss_and_grad_norm(history: dict, output_dir: str) -> None:
    """Save smoothed training loss and gradient norm curves."""
    loss_hist = history["loss"]
    grad_hist = history["grad_norm"]

    window = min(50, len(loss_hist))
    smoothed_loss = np.convolve(loss_hist, np.ones(window) / window, mode="valid")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.plot(smoothed_loss, color="steelblue", linewidth=1.5)
    ax.set_title("Smoothed Training Loss vs. Steps", fontsize=16)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_ylim(bottom=0)
    ax.grid(True)

    ax = axes[1]
    ax.plot(grad_hist, color="darkorange", linewidth=1.0, alpha=0.8)
    ax.set_title("Gradient Norm vs. Steps", fontsize=16)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("L2 Norm of Gradients")
    ax.set_yscale("log")
    ax.grid(True)

    plt.tight_layout()
    path = os.path.join(output_dir, "loss_and_grad_norm.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_gate_values(model: GIHyperTransformer, output_dir: str) -> None:
    """Save bar chart of learned gate values per layer per memory stream."""
    n_layers = len(model.layers)
    n_ext = model.n_ext

    # gate_values[layer][stream] = sigmoid(gate_logit)
    gate_matrix = np.zeros((n_layers, n_ext))

    print("\n--- Learned gate values (sigmoid) per layer ---")
    for i, layer in enumerate(model.layers):
        with torch.no_grad():
            g1 = torch.sigmoid(layer.conn1.gate_logit).cpu().numpy()
            g2 = torch.sigmoid(layer.conn2.gate_logit).cpu().numpy()
            # Average over attn and FFN connections
            g_avg = (g1 + g2) / 2.0
        gate_matrix[i] = g_avg
        vals = ", ".join(f"stream{j}={g_avg[j]:.4f}" for j in range(n_ext))
        print(f"  Layer {i+1:2d}: {vals}")

    fig, ax = plt.subplots(figsize=(max(8, n_layers * 0.6), 5))

    x = np.arange(n_layers)
    bar_width = 0.8 / max(n_ext, 1)

    for j in range(n_ext):
        offset = (j - n_ext / 2 + 0.5) * bar_width
        ax.bar(x + offset, gate_matrix[:, j], width=bar_width, label=f"Memory stream {j}")

    ax.set_xlabel("Layer Index", fontsize=13)
    ax.set_ylabel("Gate value  g = σ(gate_logit)", fontsize=13)
    ax.set_title(
        "Learned Gate Values per Layer\n"
        "(higher → stronger pull toward persistent memory prior)",
        fontsize=14,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in range(n_layers)])
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="g=0.5")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.4)

    plt.tight_layout()
    path = os.path.join(output_dir, "gate_values.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train GI-mHC (Gated Injection Hyper-Connections)"
    )
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--lr", type=float, default=8.6e-4,
        help="Base learning rate",
    )
    parser.add_argument(
        "--steps", type=int, default=10000,
        help="Total training steps",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=2000,
        help="LR warmup steps",
    )
    parser.add_argument(
        "--n-layers", type=int, default=12,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--n-tok", type=int, default=4,
        help="Number of token residual streams (default: 4)",
    )
    parser.add_argument(
        "--n-ext", type=int, default=1,
        help="Number of persistent memory streams (default: 1)",
    )
    parser.add_argument(
        "--dataset", type=str, default="Trelis/tiny-shakespeare",
        help="HuggingFace dataset ID (or 'synthetic')",
    )
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument(
        "--seq-len", type=int, default=LM_SEQ_LEN,
        help=f"Sequence length (default: {LM_SEQ_LEN})",
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    random.seed(42)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if args.dataset == "synthetic":
        seq_len = MAX_SEQ_LEN
        train_loader, _, tokenizer, pad_idx = create_dataloaders(
            MAX_SEQ_LEN, args.batch_size
        )
        print("Using synthetic copy dataset (full)")
    else:
        seq_len = args.seq_len
        dataset_id = (
            "Salesforce/wikitext" if args.dataset == "wikitext" else args.dataset
        )
        dataset_config = args.dataset_config
        if args.dataset == "wikitext" and not dataset_config:
            dataset_config = "wikitext-2-raw-v1"
        train_loader, tokenizer, pad_idx = create_lm_dataloader_full(
            dataset_id,
            dataset_config=dataset_config,
            seq_len=seq_len,
            batch_size=args.batch_size,
        )
        print(f"Using HF dataset (full): {dataset_id}")

    print(
        f"Vocabulary size: {tokenizer.vocab_size}, "
        f"batch_size={args.batch_size}, "
        f"n_tok={args.n_tok}, n_ext={args.n_ext}"
    )

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    model_config = build_gimhc_config(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=seq_len,
        n_tok=args.n_tok,
        n_ext=args.n_ext,
        n_layers=args.n_layers,
    )
    model = GIHyperTransformer(model_config).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"GI-mHC model: {n_params:,} parameters, "
        f"{args.n_layers} layers, "
        f"{args.n_tok} token streams + {args.n_ext} memory stream(s)"
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-20,
        weight_decay=0.1,
    )

    total_steps = args.steps
    scheduler = get_deepseek_step_scheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
    )

    print(
        f"AdamW betas=(0.9,0.95) eps=1e-20 wd=0.1, "
        f"DeepSeek Step LR warmup={args.warmup_steps} total={total_steps}"
    )

    os.makedirs(args.output_dir, exist_ok=True)

    batches_per_epoch = len(train_loader)
    n_epochs = max(20, (total_steps + batches_per_epoch - 1) // batches_per_epoch)
    history = train(
        model,
        train_loader,
        optimizer,
        criterion,
        n_epochs=n_epochs,
        device=device,
        scheduler=scheduler,
        max_grad_norm=1.0,
        scheduler_step_per_batch=True,
        max_steps=total_steps,
    )

    # --- Visualizations ---
    plot_loss_and_grad_norm(history, args.output_dir)
    plot_gate_values(model, args.output_dir)

    # --- Save checkpoint ---
    if args.save_model:
        ckpt_path = os.path.join(args.output_dir, "gimhc_best.pt")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": model_config,
            },
            ckpt_path,
        )
        print(f"Saved checkpoint to {ckpt_path}")

    # --- Generation demo ---
    if args.dataset == "synthetic":
        test_prompts = ["<s>ZXV", "<s>QWERTY", "<s>MNBVC"]
    else:
        test_prompts = ["First Citizen:", "ROMEO:\n", "To be or"]
    print("\n--- Generation demo ---")
    for prompt in test_prompts:
        generated = autoregressive_generate(
            model, prompt, seq_len, tokenizer, device
        )
        print(f"  Prompt: {repr(prompt)}")
        print(f"  Output: {repr(generated[:200])}")


if __name__ == "__main__":
    main()
