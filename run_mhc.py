"""Main script to run mHC experiment: train standard, HC, and mHC models."""

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
    create_lm_dataloaders,
)
from mhc.models import HyperTransformer, build_model_config
from mhc.train_utils import autoregressive_generate, evaluate, train


def get_amax_gain(matrix: torch.Tensor) -> tuple:
    """Calculates forward and backward Amax gain magnitude."""
    forward_gain = torch.max(torch.abs(matrix.sum(dim=-1))).item()
    backward_gain = torch.max(torch.abs(matrix.sum(dim=-2))).item()
    return forward_gain, backward_gain


def create_visualizations(results, sample_input, test_loader, device, output_dir):
    """Create and save all visualization plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1 & 2: Training Loss and Gradient Norm
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    for conn_type, res in results.items():
        loss_hist = res["history"]["loss"]
        smoothed_loss = np.convolve(loss_hist, np.ones(50) / 50, mode="valid")
        ax.plot(smoothed_loss, label=f"{conn_type.upper()}")
    ax.set_title("Smoothed Training Loss vs. Steps", fontsize=16)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend()
    ax.grid(True)
    ax.set_ylim(bottom=0)

    ax = axes[1]
    for conn_type, res in results.items():
        ax.plot(res["history"]["grad_norm"], label=f"{conn_type.upper()}")
    ax.set_title("Gradient Norm vs. Steps", fontsize=16)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("L2 Norm of Gradients")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_and_grad_norm.png"), dpi=150)
    plt.close()

    # Plot 3: Amax Gain Magnitude (HC and mHC only)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    plot_data = {}

    with torch.no_grad():
        for conn_type in ["hc", "mhc"]:
            model = results[conn_type]["model"]
            model.eval()
            gains = []

            x = model.embedding(sample_input)
            x = model.pos_encoder(x)
            x = model.dropout(x)
            B, S = x.shape[:2]
            x = model.initial_proj(x).view(B, S, model.config["n"], model.config["C"])

            for layer in model.layers:
                conn_module1 = layer.connection1
                x_flat = x.view(B, S, -1)
                x_normed = conn_module1.norm(x_flat)
                H_res1_unconstrained = (
                    conn_module1.a_res
                    * conn_module1.phi_res(x_normed).view(
                        B, S, model.config["n"], model.config["n"]
                    )
                    + conn_module1.b_res
                )
                if conn_type == "mhc":
                    H_res1 = conn_module1.sinkhorn(H_res1_unconstrained)
                else:
                    H_res1 = torch.tanh(H_res1_unconstrained)
                gains.append(get_amax_gain(H_res1.mean(dim=(0, 1))))

                attn_sublayer = lambda h_in: layer.attn(
                    h_in, h_in, h_in, attn_mask=None, need_weights=False
                )[0]
                x = conn_module1(x, attn_sublayer)

                conn_module2 = layer.connection2
                x_flat = x.view(B, S, -1)
                x_normed = conn_module2.norm(x_flat)
                H_res2_unconstrained = (
                    conn_module2.a_res
                    * conn_module2.phi_res(x_normed).view(
                        B, S, model.config["n"], model.config["n"]
                    )
                    + conn_module2.b_res
                )
                if conn_type == "mhc":
                    H_res2 = conn_module2.sinkhorn(H_res2_unconstrained)
                else:
                    H_res2 = torch.tanh(H_res2_unconstrained)
                gains.append(get_amax_gain(H_res2.mean(dim=(0, 1))))
                x = conn_module2(x, layer.ffn)

            plot_data[conn_type] = gains

    for i, conn_type in enumerate(["hc", "mhc"]):
        ax = axes[i]
        gains = plot_data[conn_type]
        labels = [
            f'L{l//2+1}_{"Attn" if l%2==0 else "FFN"}' for l in range(len(gains))
        ]
        forward_gains = [g[0] for g in gains]
        backward_gains = [g[1] for g in gains]
        x_pos = np.arange(len(labels))
        ax.bar(x_pos - 0.2, forward_gains, width=0.4, label="Forward Gain")
        ax.bar(x_pos + 0.2, backward_gains, width=0.4, label="Backward Gain")
        ax.axhline(1.0, color="r", linestyle="--", label="Ideal (1.0)")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_title(f"Amax Gain Magnitude ({conn_type.upper()})", fontsize=16)
        ax.set_ylabel("Gain Magnitude")
        ax.legend()
        ax.grid(axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "amax_gain.png"), dpi=150)
    plt.close()

    # Plot 4: Composite H_res Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    with torch.no_grad():
        for i, conn_type in enumerate(["hc", "mhc"]):
            model = results[conn_type]["model"]
            model.eval()

            composite_H_res = (
                torch.eye(model.config["n"], device=device).unsqueeze(0).unsqueeze(0)
            )

            x = model.embedding(sample_input)
            x = model.pos_encoder(x)
            x = model.dropout(x)
            B, S = x.shape[:2]
            x = model.initial_proj(x).view(
                B, S, model.config["n"], model.config["C"]
            )

            for layer in model.layers:
                conn_module1 = layer.connection1
                x_flat1 = x.view(B, S, -1)
                x_normed1 = conn_module1.norm(x_flat1)
                H_res1_unconstrained = (
                    conn_module1.a_res
                    * conn_module1.phi_res(x_normed1).view(
                        B, S, model.config["n"], model.config["n"]
                    )
                    + conn_module1.b_res
                )
                if conn_type == "mhc":
                    H_res1 = conn_module1.sinkhorn(H_res1_unconstrained)
                else:
                    H_res1 = torch.tanh(H_res1_unconstrained)
                composite_H_res = torch.einsum(
                    "bsij,bsjk->bsik", H_res1, composite_H_res
                )
                attn_sublayer = lambda h_in: layer.attn(
                    h_in, h_in, h_in, attn_mask=None, need_weights=False
                )[0]
                x = conn_module1(x, attn_sublayer)

                conn_module2 = layer.connection2
                x_flat2 = x.view(B, S, -1)
                x_normed2 = conn_module2.norm(x_flat2)
                H_res2_unconstrained = (
                    conn_module2.a_res
                    * conn_module2.phi_res(x_normed2).view(
                        B, S, model.config["n"], model.config["n"]
                    )
                    + conn_module2.b_res
                )
                if conn_type == "mhc":
                    H_res2 = conn_module2.sinkhorn(H_res2_unconstrained)
                else:
                    H_res2 = torch.tanh(H_res2_unconstrained)
                composite_H_res = torch.einsum(
                    "bsij,bsjk->bsik", H_res2, composite_H_res
                )
                x = conn_module2(x, layer.ffn)

            avg_composite_H = composite_H_res.mean(dim=(0, 1)).cpu().numpy()

            ax = axes[i]
            im = ax.imshow(avg_composite_H, cmap="viridis")
            fig.colorbar(im, ax=ax)
            ax.set_title(f"Composite H_res ({conn_type.upper()})", fontsize=14)
            ax.set_xlabel("Input Stream")
            ax.set_ylabel("Output Stream")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "composite_H_res.png"), dpi=150)
    plt.close()

    print(f"\nPlots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Run mHC experiment")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save plots (default: output)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively (in addition to saving)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50; mHC benefits from longer training)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=6e-4,
        help="Learning rate (default: 6e-4, tokenbender-style)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Trelis/tiny-shakespeare",
        help="Dataset (default: Trelis/tiny-shakespeare)",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Dataset config for multi-config datasets (e.g. wikitext-2-raw-v1 for wikitext)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=LM_SEQ_LEN,
        help=f"Sequence length for LM datasets (default: {LM_SEQ_LEN})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})",
    )
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Prefer CUDA, then MPS (Apple Silicon), then CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Using device: {device}")

    # Create dataloaders, tokenizer, and pad index
    if args.dataset == "synthetic":
        seq_len = MAX_SEQ_LEN
        train_loader, test_loader, tokenizer, pad_idx = create_dataloaders(
            MAX_SEQ_LEN, args.batch_size
        )
        print("Using synthetic copy dataset")
    else:
        seq_len = args.seq_len
        dataset_id = args.dataset
        dataset_config = args.dataset_config
        train_loader, test_loader, tokenizer, pad_idx = create_lm_dataloaders(
            dataset_id,
            dataset_config=dataset_config,
            seq_len=seq_len,
            batch_size=args.batch_size,
        )
        print(
            f"Using Hugging Face dataset: {dataset_id}"
            + (f" ({dataset_config})" if dataset_config else "")
        )

    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Sequence length: {seq_len}")

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # Base config
    base_config = build_model_config(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=seq_len,
        connection_type="mhc",  # overwritten per model
    )

    # Train all three connection types
    connection_types = ["standard", "hc", "mhc"]
    results = {}

    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(200, total_steps // 10)

    for conn_type in connection_types:
        print(f"\n{'='*20} Training {conn_type.upper()} Model {'='*20}")

        model_config = base_config.copy()
        model_config["connection_type"] = conn_type

        # mHC regularization: higher dropout, shallower on small datasets
        if conn_type == "mhc":
            model_config["dropout"] = 0.3
            if args.dataset == "synthetic":
                model_config["n_layers"] = 4

        model = HyperTransformer(model_config).to(device)

        lr = args.lr * 0.5 if conn_type == "mhc" else args.lr
        weight_decay = 0.25 if conn_type == "mhc" else 0.1
        optimizer = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        if conn_type == "mhc":
            print(f"  mHC regularization: dropout={model_config['dropout']}, lr={lr:.2e}, wd={weight_decay}, n_layers={model_config['n_layers']}")
        # Warmup + cosine decay (tokenbender-style)
        warmup = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
        )
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=6e-5,
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, [warmup, cosine], milestones=[warmup_steps]
        )

        history = train(
            model,
            train_loader,
            optimizer,
            criterion,
            args.epochs,
            device,
            scheduler=scheduler,
            max_grad_norm=1.0,
            scheduler_step_per_batch=True,
        )

        test_loss, test_acc, test_em = evaluate(
            model, test_loader, criterion, pad_idx, device
        )

        results[conn_type] = {
            "model": model,
            "history": history,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "test_exact_match": test_em,
        }

        print(f"\nResults for {conn_type.upper()}:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Test Exact Match: {test_em:.4f}")

    # Summary table
    print("\n\n--- Final Results Summary ---")
    print(f"| {'Model':<10} | {'Test Loss':<10} | {'Accuracy':<10} | {'Exact Match':<12} |")
    print(f"|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*14}|")
    for conn_type, res in results.items():
        print(
            f"| {conn_type:<10} | {res['test_loss']:<10.4f} | "
            f"{res['test_accuracy']:<10.4f} | {res['test_exact_match']:<12.4f} |"
        )

    # Generation demos
    if args.dataset == "synthetic":
        test_prompts = ["<s>ZXV", "<s>QWERTY", "<s>MNBVC"]
    else:
        test_prompts = ["First Citizen:", "ROMEO:\n", "To be or"]
    for conn_type, res in results.items():
        print(f"\n--- Generating with {conn_type.upper()} model ---")
        model = res["model"]
        for prompt in test_prompts:
            generated = autoregressive_generate(
                model, prompt, seq_len, tokenizer, device
            )
            print(f"  Prompt: {repr(prompt)}")
            print(f"  Output: {repr(generated[:200])}")

    # Visualizations
    sample_input = next(iter(test_loader))["input_ids"][:1].to(device)
    create_visualizations(
        results, sample_input, test_loader, device, args.output_dir
    )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
