"""Training, evaluation, and generation utilities."""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


def get_deepseek_step_scheduler(
    optimizer: optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    decay_step_ratio: Tuple[float, float] = (0.8, 0.9),
    decay_rate: Tuple[float, float] = (0.316, 0.1),
):
    """DeepSeek-style Step LR: linear warmup then step decay at 0.8*T and 0.9*T."""
    milestones = [
        int(total_steps * decay_step_ratio[0]),
        int(total_steps * decay_step_ratio[1]),
    ]

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        lr_mult = 1.0
        if step >= milestones[1]:
            lr_mult = decay_rate[0] * decay_rate[1]
        elif step >= milestones[0]:
            lr_mult = decay_rate[0]
        return lr_mult

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    n_epochs: int,
    device: torch.device,
    scheduler=None,
    max_grad_norm: float = 1.0,
    scheduler_step_per_batch: bool = False,
    max_steps: Optional[int] = None,
) -> Dict[str, List[float]]:
    """Trains the model. If max_steps is set, stops after that many optimizer steps."""
    model.train()
    history = defaultdict(list)
    global_step = 0

    for epoch in range(n_epochs):
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)

            logits = model(input_ids)
            loss = criterion(
                logits.view(-1, model.config["vocab_size"]), target_ids.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=max_grad_norm
            )

            optimizer.step()
            if scheduler is not None and scheduler_step_per_batch:
                scheduler.step()

            history["loss"].append(loss.item())
            history["grad_norm"].append(grad_norm.item())
            global_step += 1

            if max_steps is not None and global_step >= max_steps:
                print(
                    f"Step {global_step}/{max_steps}, Loss: {loss.item():.4f}, "
                    f"Grad Norm: {grad_norm.item():.4f}"
                )
                return history

        if scheduler is not None and not scheduler_step_per_batch:
            scheduler.step()

        print(
            f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}, "
            f"Grad Norm: {grad_norm.item():.4f}"
        )

    return history


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    pad_idx: int,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Evaluates the model on a given dataset."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_elements = 0
    exact_matches = 0
    total_sequences = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)

            logits = model(input_ids)
            loss = criterion(
                logits.view(-1, model.config["vocab_size"]), target_ids.view(-1)
            )
            total_loss += loss.item() * input_ids.size(0)

            preds = logits.argmax(dim=-1)

            non_pad_mask = target_ids != pad_idx
            total_correct += (preds[non_pad_mask] == target_ids[non_pad_mask]).sum().item()
            total_elements += non_pad_mask.sum().item()

            seq_len = non_pad_mask.sum(dim=1)
            correct_tokens_per_seq = (
                (preds == target_ids).logical_and(non_pad_mask).sum(dim=1)
            )
            exact_matches += (correct_tokens_per_seq == seq_len).sum().item()
            total_sequences += input_ids.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_correct / total_elements
    exact_match_rate = exact_matches / total_sequences
    return avg_loss, accuracy, exact_match_rate


def autoregressive_generate(
    model: nn.Module,
    prompt: str,
    max_len: int,
    tokenizer,
    device: torch.device,
) -> str:
    """Generates a sequence auto-regressively."""
    model.eval()
    tokens = tokenizer.encode(prompt)
    input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)

    eos_idx = tokenizer.char_to_idx.get("</s>", None)

    for _ in range(max_len - len(tokens)):
        with torch.no_grad():
            logits = model(input_tensor)
        next_token_logits = logits[0, -1, :]
        next_token = torch.argmax(next_token_logits).item()

        input_tensor = torch.cat(
            [input_tensor, torch.tensor([[next_token]], device=device)], dim=1
        )

        if eos_idx is not None and next_token == eos_idx:
            break

    return tokenizer.decode(input_tensor.squeeze().tolist())
