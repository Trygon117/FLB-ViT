import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def add_gaussian_noise(images: torch.Tensor, sigma: float, seed: int) -> torch.Tensor:
    if sigma <= 0.0:
        return images
    gen = torch.Generator(device=images.device)
    gen.manual_seed(seed)
    noise = torch.randn(
        images.shape, generator=gen, device=images.device, dtype=images.dtype
    ) * sigma
    return images + noise


def compute_stream_deltas(
    model: nn.Module,
    final_forward: torch.Tensor,
    prev_forward: torch.Tensor,
    next_lateral: List[torch.Tensor],
    prev_lateral: List[torch.Tensor],
    next_backward: List[torch.Tensor],
    prev_backward: List[torch.Tensor],
    next_top_backward: torch.Tensor,
    prev_top_backward: torch.Tensor
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    l_deltas = [
        model.compute_relative_delta(next_lateral[i], prev_lateral[i])
        for i in range(model.depth)
    ]
    b_deltas = [
        model.compute_relative_delta(next_backward[i], prev_backward[i])
        for i in range(model.depth)
    ]
    f_delta = model.compute_relative_delta(final_forward, prev_forward)
    d_delta = model.compute_relative_delta(next_top_backward, prev_top_backward)
    all_deltas = torch.stack(l_deltas + b_deltas + [f_delta, d_delta], dim=0)
    image_max_delta = all_deltas.max(dim=0).values
    return l_deltas, b_deltas, f_delta, d_delta, image_max_delta


def load_all_histories(
    checkpoint_root: str = "./checkpoints",
    include_models: Optional[List[str]] = None
) -> Dict[str, dict]:
    histories = {}
    if not os.path.exists(checkpoint_root):
        print(f"Directory {checkpoint_root} does not exist.")
        return histories

    folder_names = sorted(os.listdir(checkpoint_root))
    if include_models is not None:
        folder_names = [f for f in folder_names if f in include_models]

    print(
        f"{'Model Name':<28} | {'Params':>10} | {'Sec/Ep':>7} | "
        f"{'Train Acc':>9} | {'Peak Val Acc':>12} | {'Final Val Acc':>13}"
    )
    print("-" * 94)

    for folder in folder_names:
        hist_path = os.path.join(checkpoint_root, folder, "history.json")
        if os.path.isfile(hist_path):
            with open(hist_path, "r") as f:
                h = json.load(f)
            histories[folder] = h

            params = h.get("total_params", 0)
            avg_sec = np.mean(h["elapsed_sec"]) if h.get("elapsed_sec") else 0.0
            final_train = h["train_acc"][-1] if h.get("train_acc") else 0.0
            peak_val = max(h["val_acc"]) if h.get("val_acc") else 0.0
            final_val = h["val_acc"][-1] if h.get("val_acc") else 0.0

            print(
                f"{folder:<28} | {params:10,d} | {avg_sec:6.1f}s | "
                f"{final_train:8.2f}% | {peak_val:11.2f}% | {final_val:12.2f}%"
            )

    return histories


@torch.no_grad()
def _eval_flb_batch_multi_horizon(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    max_eval_steps: int = 50,
    trained_limit: int = 12,
    threshold: float = 1e-2
):
    batch_size = images.size(0)
    device = images.device
    min_check_step = model.depth - 1

    input_tokens = model.patch_embed(images)
    lateral_states, backward_states, top_backward = model.init_hidden_states(batch_size, device)
    prev_forward = None

    step_correct = torch.zeros(max_eval_steps, dtype=torch.long, device=device)

    converged_12 = torch.zeros(batch_size, dtype=torch.bool, device=device)
    steps_12 = torch.zeros(batch_size, dtype=torch.long, device=device)
    halted_preds_12 = torch.zeros(batch_size, dtype=torch.long, device=device)

    converged_50 = torch.zeros(batch_size, dtype=torch.bool, device=device)
    steps_50 = torch.zeros(batch_size, dtype=torch.long, device=device)
    halted_preds_50 = torch.zeros(batch_size, dtype=torch.long, device=device)

    for step in range(max_eval_steps):
        final_forward, next_lateral, next_backward, next_top_backward, _ = model.forward_tick(
            input_tokens, lateral_states, backward_states, top_backward, with_aux=False
        )
        preds = model.decoder.classify(next_top_backward).argmax(dim=-1)
        step_correct[step] = (preds == labels).sum()

        if step > 0:
            _, _, _, _, image_max_delta = compute_stream_deltas(
                model, final_forward, prev_forward,
                next_lateral, lateral_states,
                next_backward, backward_states,
                next_top_backward, top_backward
            )
        else:
            image_max_delta = torch.full((batch_size,), float("inf"), device=device)

        if step < trained_limit:
            still_running_12 = ~converged_12
            steps_12 += still_running_12.long()
            halted_preds_12 = torch.where(still_running_12, preds, halted_preds_12)
            if threshold > 0.0 and step >= min_check_step:
                converged_12 = converged_12 | (image_max_delta < threshold)

        still_running_50 = ~converged_50
        steps_50 += still_running_50.long()
        halted_preds_50 = torch.where(still_running_50, preds, halted_preds_50)
        if threshold > 0.0 and step >= min_check_step:
            converged_50 = converged_50 | (image_max_delta < threshold)

        lateral_states = next_lateral
        backward_states = next_backward
        top_backward = next_top_backward
        prev_forward = final_forward

    correct_adapt_12 = (halted_preds_12 == labels).sum().item()
    correct_adapt_50 = (halted_preds_50 == labels).sum().item()

    return (
        step_correct.cpu().numpy(),
        correct_adapt_12,
        steps_12.float().sum().item(),
        correct_adapt_50,
        steps_50.float().sum().item()
    )


@torch.no_grad()
def evaluate_noise_robustness(
    flb_model: nn.Module,
    baseline_models: Dict[str, nn.Module],
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    noise_levels: List[float] = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30],
    max_eval_steps: int = 50,
    trained_limit: int = 12,
    threshold: float = 1e-2
) -> dict:
    flb_model.eval()
    for m in baseline_models.values():
        m.eval()

    results = {
        "noise_levels": noise_levels,
        "max_eval_steps": max_eval_steps,
        "trained_limit": trained_limit,
        "flb_step1_acc": [],
        "flb_fixed12_acc": [],
        "flb_adapt12_acc": [],
        "flb_adapt12_steps": [],
        "flb_fixed50_acc": [],
        "flb_adapt50_acc": [],
        "flb_adapt50_steps": [],
        "flb_step_curves": {},
        "baselines": {name: [] for name in baseline_models.keys()}
    }

    for sigma in noise_levels:
        total_samples = 0
        flb_step_correct_total = np.zeros(max_eval_steps, dtype=np.int64)
        flb_adapt12_correct = 0
        flb_adapt12_steps_sum = 0.0
        flb_adapt50_correct = 0
        flb_adapt50_steps_sum = 0.0
        base_correct = {name: 0 for name in baseline_models.keys()}

        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            total_samples += images.size(0)

            seed = 10000 * int(round(sigma * 1000)) + batch_idx
            noisy_images = add_gaussian_noise(images, sigma, seed)

            step_corr, c_a12, s_a12, c_a50, s_a50 = _eval_flb_batch_multi_horizon(
                flb_model, noisy_images, labels,
                max_eval_steps=max_eval_steps,
                trained_limit=trained_limit,
                threshold=threshold
            )
            flb_step_correct_total += step_corr
            flb_adapt12_correct += c_a12
            flb_adapt12_steps_sum += s_a12
            flb_adapt50_correct += c_a50
            flb_adapt50_steps_sum += s_a50

            for name, b_model in baseline_models.items():
                b_preds = b_model(noisy_images).argmax(dim=-1)
                base_correct[name] += (b_preds == labels).sum().item()

        step_acc_curve = 100.0 * flb_step_correct_total / total_samples
        results["flb_step_curves"][sigma] = step_acc_curve
        results["flb_step1_acc"].append(float(step_acc_curve[0]))
        results["flb_fixed12_acc"].append(float(step_acc_curve[trained_limit - 1]))
        results["flb_adapt12_acc"].append(100.0 * flb_adapt12_correct / total_samples)
        results["flb_adapt12_steps"].append(flb_adapt12_steps_sum / total_samples)
        results["flb_fixed50_acc"].append(float(step_acc_curve[max_eval_steps - 1]))
        results["flb_adapt50_acc"].append(100.0 * flb_adapt50_correct / total_samples)
        results["flb_adapt50_steps"].append(flb_adapt50_steps_sum / total_samples)

        for name in baseline_models.keys():
            results["baselines"][name].append(100.0 * base_correct[name] / total_samples)

        print(
            f"Noise std={sigma:.2f} | "
            f"FLB Step1: {results['flb_step1_acc'][-1]:.2f}% | "
            f"FLB<=12: {results['flb_adapt12_acc'][-1]:.2f}% ({results['flb_adapt12_steps'][-1]:.2f} stp) | "
            f"FLB<=50: {results['flb_adapt50_acc'][-1]:.2f}% ({results['flb_adapt50_steps'][-1]:.2f} stp) | "
            f"FLB@50: {results['flb_fixed50_acc'][-1]:.2f}%"
        )

    print("\n" + "=" * 98)
    header = f"{'Model / Mode':<30} | " + " | ".join([f"std={s:.2f}" for s in noise_levels])
    print(header)
    print("-" * 98)

    flb_rows = [
        ("FLBViT (Step 1 - Feedforward)", results["flb_step1_acc"]),
        ("FLBViT (Adaptive <= 12 Steps)", results["flb_adapt12_acc"]),
        ("FLBViT (Fixed 12 Steps)", results["flb_fixed12_acc"]),
        ("FLBViT (Adaptive <= 50 Steps)", results["flb_adapt50_acc"]),
        ("FLBViT (Fixed 50 Steps)", results["flb_fixed50_acc"]),
    ]
    for row_name, vals in flb_rows:
        print(f"{row_name:<30} | " + " | ".join([f"{v:7.2f}%" for v in vals]))

    print("-" * 98)
    for name, vals in results["baselines"].items():
        print(f"{name:<30} | " + " | ".join([f"{v:7.2f}%" for v in vals]))
    print("=" * 98)

    return results


@torch.no_grad()
def evaluate_contraction_and_gates(
    flb_model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_steps: int = 20,
    num_batches: Optional[int] = 20
) -> dict:
    if hasattr(flb_model, "forward_tick") and hasattr(flb_model.__class__, "forward_tick"):
        flb_model.forward_tick = flb_model.__class__.forward_tick.__get__(flb_model, flb_model.__class__)
    flb_model.eval()

    depth = flb_model.depth
    lat_deltas_batches = [[] for _ in range(depth)]
    back_deltas_batches = [[] for _ in range(depth)]
    fwd_deltas_batches, dec_deltas_batches, max_deltas_batches, step_acc_batches = [], [], [], []
    gate_batches = {i: {"f": [], "z": [], "s": []} for i in range(depth)}

    for b_idx, (images, labels) in enumerate(test_loader):
        if num_batches is not None and b_idx >= num_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        bsz = images.size(0)

        batch_gate_logs = {i: {"f": [], "z": [], "s": []} for i in range(depth)}
        hooks = []

        def make_hook(l_idx, g_name):
            def hook(module, inp, out):
                batch_gate_logs[l_idx][g_name].append(torch.sigmoid(out.detach()).mean().item())
            return hook

        for i, block in enumerate(flb_model.blocks):
            hooks.append(block.gate_f.register_forward_hook(make_hook(i, "f")))
            hooks.append(block.gate_z.register_forward_hook(make_hook(i, "z")))
            hooks.append(block.gate_s.register_forward_hook(make_hook(i, "s")))

        input_tokens = flb_model.patch_embed(images)
        lateral_states, backward_states, top_backward = flb_model.init_hidden_states(bsz, device)
        prev_forward = None

        b_lat = [[] for _ in range(depth)]
        b_back = [[] for _ in range(depth)]
        b_fwd, b_dec, b_max, b_acc = [], [], [], []

        for step in range(num_steps):
            final_forward, next_lateral, next_backward, next_top_backward, _ = flb_model.forward_tick(
                input_tokens, lateral_states, backward_states, top_backward, with_aux=False
            )
            preds = flb_model.decoder.classify(next_top_backward).argmax(dim=-1)
            b_acc.append(100.0 * (preds == labels).float().mean().item())

            if step > 0:
                l_d, b_d, f_d, d_d, img_max = compute_stream_deltas(
                    flb_model, final_forward, prev_forward,
                    next_lateral, lateral_states,
                    next_backward, backward_states,
                    next_top_backward, top_backward
                )
                for i in range(depth):
                    b_lat[i].append(l_d[i].mean().item())
                    b_back[i].append(b_d[i].mean().item())
                b_fwd.append(f_d.mean().item())
                b_dec.append(d_d.mean().item())
                b_max.append(img_max.mean().item())

            lateral_states = next_lateral
            backward_states = next_backward
            top_backward = next_top_backward
            prev_forward = final_forward

        for h in hooks:
            h.remove()

        for i in range(depth):
            lat_deltas_batches[i].append(b_lat[i])
            back_deltas_batches[i].append(b_back[i])
            for g_name in ["f", "z", "s"]:
                gate_batches[i][g_name].append(batch_gate_logs[i][g_name])

        fwd_deltas_batches.append(b_fwd)
        dec_deltas_batches.append(b_dec)
        max_deltas_batches.append(b_max)
        step_acc_batches.append(b_acc)

    lat_mean, lat_std = np.mean(lat_deltas_batches, axis=1), np.std(lat_deltas_batches, axis=1)
    back_mean, back_std = np.mean(back_deltas_batches, axis=1), np.std(back_deltas_batches, axis=1)
    fwd_mean, fwd_std = np.mean(fwd_deltas_batches, axis=0), np.std(fwd_deltas_batches, axis=0)
    dec_mean, dec_std = np.mean(dec_deltas_batches, axis=0), np.std(dec_deltas_batches, axis=0)
    max_mean, max_std = np.mean(max_deltas_batches, axis=0), np.std(max_deltas_batches, axis=0)
    acc_mean = np.mean(step_acc_batches, axis=0)
    n_run = len(fwd_deltas_batches)

    print(f"Averaged Contraction & Gate Dynamics over {n_run} batches:")
    print(f"{'Step':>4} | {'L0':>6} {'L1':>6} {'L2':>6} | {'B0':>6} {'B1':>6} {'B2':>6} | {'Fwd':>6} {'Dec':>6} | {'MeanMax':>7} | {'Acc':>6}")
    print("-" * 86)
    print(f"{1:4d} | {'--- Initial Feedforward Sweep (Step 0) ---':^51} | {'---':>7} | {acc_mean[0]:5.2f}%")
    for s in range(num_steps - 1):
        print(
            f"{s + 2:4d} | {lat_mean[0, s]:6.4f} {lat_mean[1, s]:6.4f} {lat_mean[2, s]:6.4f} | "
            f"{back_mean[0, s]:6.4f} {back_mean[1, s]:6.4f} {back_mean[2, s]:6.4f} | "
            f"{fwd_mean[s]:6.4f} {dec_mean[s]:6.4f} | {max_mean[s]:7.4f} | {acc_mean[s + 1]:5.2f}%"
        )

    print("\nMean Sigmoid Gate Activations Across Batches [1.0 = keep input, 0.0 = write candidate]:")
    print("-" * 86)
    for step in [0, 1, 2, 3, 5, 11]:
        if step < num_steps:
            row_str = f"Step {step + 1:2d} | "
            for i in range(depth):
                f_m = np.mean(gate_batches[i]["f"], axis=0)[step]
                z_m = np.mean(gate_batches[i]["z"], axis=0)[step]
                s_m = np.mean(gate_batches[i]["s"], axis=0)[step]
                row_str += f"L{i}[f:{f_m:.3f} z:{z_m:.3f} s:{s_m:.3f}] | "
            print(row_str)

    return {
        "num_steps": num_steps,
        "num_batches_run": n_run,
        "lat_mean": lat_mean,
        "lat_std": lat_std,
        "back_mean": back_mean,
        "back_std": back_std,
        "fwd_mean": fwd_mean,
        "fwd_std": fwd_std,
        "dec_mean": dec_mean,
        "dec_std": dec_std,
        "max_mean": max_mean,
        "max_std": max_std,
        "acc_mean": acc_mean,
        "gate_batches": gate_batches
    }


def _compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            ece += np.abs(np.mean(accuracies[in_bin]) - np.mean(confidences[in_bin])) * prop_in_bin
    return float(ece * 100.0)


@torch.no_grad()
def evaluate_reliability_and_subgroups(
    flb_model: nn.Module,
    baseline_models: Dict[str, nn.Module],
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    noise_levels: List[float] = [0.0, 0.15, 0.30],
    trained_limit: int = 12
) -> dict:
    flb_model.eval()
    for m in baseline_models.values():
        m.eval()

    reliability = {sigma: {} for sigma in noise_levels}
    per_class_gains = {}

    for sigma in noise_levels:
        all_labels = []
        flb_s1_probs, flb_s12_probs = [], []
        base_probs = {name: [] for name in baseline_models.keys()}

        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device, non_blocking=True)
            seed = 10000 * int(round(sigma * 1000)) + batch_idx
            noisy_images = add_gaussian_noise(images, sigma, seed)

            bsz = noisy_images.size(0)
            input_tokens = flb_model.patch_embed(noisy_images)
            lat, back, top_b = flb_model.init_hidden_states(bsz, device)

            for step in range(trained_limit):
                _, lat, back, top_b, _ = flb_model.forward_tick(input_tokens, lat, back, top_b, with_aux=False)
                if step == 0:
                    flb_s1_probs.append(F.softmax(flb_model.decoder.classify(top_b), dim=-1).cpu().numpy())
                if step == trained_limit - 1:
                    flb_s12_probs.append(F.softmax(flb_model.decoder.classify(top_b), dim=-1).cpu().numpy())

            for name, b_model in baseline_models.items():
                base_probs[name].append(F.softmax(b_model(noisy_images), dim=-1).cpu().numpy())

            all_labels.append(labels.numpy())

        y_true = np.concatenate(all_labels, axis=0)
        p_s1 = np.concatenate(flb_s1_probs, axis=0)
        p_s12 = np.concatenate(flb_s12_probs, axis=0)

        def calc_stats(p_arr):
            ent = -np.sum(p_arr * np.log(np.clip(p_arr, 1e-8, 1.0)), axis=1)
            preds = np.argmax(p_arr, axis=1)
            correct_mask = (preds == y_true)
            return {
                "ece": _compute_ece(p_arr, y_true),
                "mean_entropy": float(np.mean(ent)),
                "ent_correct": float(np.mean(ent[correct_mask])) if np.any(correct_mask) else 0.0,
                "ent_wrong": float(np.mean(ent[~correct_mask])) if np.any(~correct_mask) else 0.0,
            }

        reliability[sigma]["FLBViT (Step 1)"] = calc_stats(p_s1)
        reliability[sigma]["FLBViT (Step 12)"] = calc_stats(p_s12)
        for name in baseline_models.keys():
            reliability[sigma][name] = calc_stats(np.concatenate(base_probs[name], axis=0))

        class_gains = []
        for c in range(10):
            c_mask = (y_true == c)
            acc1 = 100.0 * np.mean(np.argmax(p_s1[c_mask], axis=1) == c)
            acc12 = 100.0 * np.mean(np.argmax(p_s12[c_mask], axis=1) == c)
            class_gains.append(acc12 - acc1)
        per_class_gains[sigma] = class_gains

    print("\n" + "=" * 92)
    print(f"{'Model':<28} | {'ECE (s=0.00)':>12} | {'ECE (s=0.15)':>12} | {'ECE (s=0.30)':>12} | {'Ent (s=0.30)':>12}")
    print("-" * 92)
    for m_name in reliability[noise_levels[0]].keys():
        e0 = reliability[noise_levels[0]][m_name]["ece"]
        e15 = reliability[noise_levels[1]][m_name]["ece"] if len(noise_levels) > 1 else 0.0
        e30 = reliability[noise_levels[-1]][m_name]["ece"]
        h30 = reliability[noise_levels[-1]][m_name]["mean_entropy"]
        print(f"{m_name:<28} | {e0:11.2f}% | {e15:11.2f}% | {e30:11.2f}% | {h30:12.3f}")
    print("=" * 92)

    return {
        "noise_levels": noise_levels,
        "reliability": reliability,
        "per_class_gains": per_class_gains
    }


@torch.no_grad()
def collect_qualitative_cases(
    flb_model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    sigma: float = 0.20,
    trained_limit: int = 12,
    num_recoveries: int = 3,
    num_failures: int = 2
) -> dict:
    flb_model.eval()
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1, 3, 1, 1)

    recoveries = []
    failures = []

    for batch_idx, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        noisy_images = add_gaussian_noise(images, sigma, seed=9999 + batch_idx)

        bsz = noisy_images.size(0)
        input_tokens = flb_model.patch_embed(noisy_images)
        lat, back, top_b = flb_model.init_hidden_states(bsz, device)

        step_probs = []
        step_attn_maps = []

        for _ in range(trained_limit):
            _, lat, back, top_b, _ = flb_model.forward_tick(input_tokens, lat, back, top_b, with_aux=False)
            probs = F.softmax(flb_model.decoder.classify(top_b), dim=-1)
            pool_w = F.softmax(flb_model.decoder.attn_pool(top_b), dim=1).view(bsz, 8, 8)
            step_probs.append(probs.cpu())
            step_attn_maps.append(pool_w.cpu())

        pred1 = step_probs[0].argmax(dim=-1)
        pred12 = step_probs[-1].argmax(dim=-1)
        y_cpu = labels.cpu()
        vis_noisy = torch.clamp(noisy_images * std + mean, 0.0, 1.0).cpu()

        for i in range(bsz):
            case = (
                vis_noisy[i], int(y_cpu[i]), int(pred1[i]), int(pred12[i]),
                [sp[i] for sp in step_probs], step_attn_maps[-1][i]
            )
            if pred1[i] != y_cpu[i] and pred12[i] == y_cpu[i] and len(recoveries) < num_recoveries:
                recoveries.append(case)
            elif pred1[i] != y_cpu[i] and pred12[i] != y_cpu[i] and step_probs[-1][i, pred12[i]] > 0.65 and len(failures) < num_failures:
                failures.append(case)

        if len(recoveries) >= num_recoveries and len(failures) >= num_failures:
            break

    return {
        "sigma": sigma,
        "trained_limit": trained_limit,
        "recoveries": recoveries,
        "failures": failures
    }