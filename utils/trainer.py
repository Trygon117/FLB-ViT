import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    is_flb: bool = False,
    max_steps: int = 12,
    threshold: float = 1e-2
):
    model.train()
    total_ce = 0.0
    total_aux = 0.0
    total_steps = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = images.size(0)

        optimizer.zero_grad()

        if is_flb:
            logits, ce_loss, aux_loss, steps = model(
                images,
                max_steps=max_steps,
                threshold=threshold,
                labels=labels,
                criterion=criterion
            )
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            ce_loss = loss.item()
            aux_loss = 0.0
            steps = 1.0

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_ce += ce_loss * batch_size
        total_aux += aux_loss * batch_size
        total_steps += steps * batch_size
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += batch_size

    avg_ce = total_ce / total
    avg_aux = total_aux / total
    acc = 100.0 * correct / total
    avg_steps = total_steps / total
    return avg_ce, avg_aux, acc, avg_steps


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_flb: bool = False,
    max_steps: int = 12,
    threshold: float = 1e-2
):
    model.eval()
    correct = 0
    total = 0
    total_steps = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = images.size(0)

        if is_flb:
            logits, steps = model(
                images, max_steps=max_steps, threshold=threshold
            )
        else:
            logits = model(images)
            steps = 1.0

        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total_steps += steps * batch_size
        total += batch_size

    acc = 100.0 * correct / total
    avg_steps = total_steps / total
    return acc, avg_steps


def fit_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    is_flb: bool = False,
    max_steps: int = 12,
    threshold: float = 1e-2,
    save_every: int = 5
):
    save_dir = os.path.join("./checkpoints", model_name)
    os.makedirs(save_dir, exist_ok=True)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'=' * 80}")
    print(f"Training {model_name} | Trainable Parameters: {total_params:,}")
    print(f"Checkpoints directory: {save_dir}")
    print(f"{'=' * 80}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    history = {
        "model_name": model_name,
        "total_params": total_params,
        "epochs": [],
        "train_ce": [],
        "train_aux": [],
        "train_acc": [],
        "train_steps": [],
        "val_acc": [],
        "val_steps": [],
        "elapsed_sec": []
    }
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_ce, train_aux, train_acc, train_steps = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            is_flb=is_flb,
            max_steps=max_steps,
            threshold=threshold
        )
        val_acc, val_steps = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            is_flb=is_flb,
            max_steps=max_steps,
            threshold=threshold
        )
        scheduler.step()
        elapsed = time.time() - start_time

        # Record metrics for every single epoch
        history["epochs"].append(epoch)
        history["train_ce"].append(train_ce)
        history["train_aux"].append(train_aux)
        history["train_acc"].append(train_acc)
        history["train_steps"].append(train_steps)
        history["val_acc"].append(val_acc)
        history["val_steps"].append(val_steps)
        history["elapsed_sec"].append(elapsed)

        if is_flb:
            print(
                f"Epoch {epoch:02d}/{epochs:02d} ({elapsed:.1f}s) | "
                f"Train CE: {train_ce:.4f} | Aux: {train_aux:.4f} | "
                f"Train Acc: {train_acc:.2f}% (Steps: {train_steps:.2f}) | "
                f"Val Acc: {val_acc:.2f}% (Steps: {val_steps:.2f})"
            )
        else:
            print(
                f"Epoch {epoch:02d}/{epochs:02d} ({elapsed:.1f}s) | "
                f"Train CE: {train_ce:.4f} | "
                f"Train Acc: {train_acc:.2f}% | "
                f"Val Acc: {val_acc:.2f}%"
            )

        # Save lightweight JSON metrics history every epoch for easy graphing
        with open(os.path.join(save_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        checkpoint = {
            "epoch": epoch,
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_ce": train_ce,
            "train_aux": train_aux,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_steps": val_steps,
            "history": history
        }

        # Always update latest.pt
        torch.save(checkpoint, os.path.join(save_dir, "latest.pt"))

        # Save periodic snapshot every `save_every` epochs (e.g., 5, 10, 15, 20)
        if save_every > 0 and (epoch % save_every == 0):
            torch.save(
                checkpoint, os.path.join(save_dir, f"epoch_{epoch:02d}.pt")
            )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, os.path.join(save_dir, "best.pt"))

    return history