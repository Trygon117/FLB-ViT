import os
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from .eval import CIFAR10_CLASSES


def save_figure(
    fig: plt.Figure,
    filename: str,
    save_dir: str = "./figures",
    bg_color: str = "white"
):
    os.makedirs(save_dir, exist_ok=True)
    pdf_path = os.path.join(save_dir, f"{filename}.pdf")
    png_path = os.path.join(save_dir, f"{filename}.png")

    fig.patch.set_facecolor(bg_color)
    for ax in fig.axes:
        if ax.patch.get_visible():
            ax.set_facecolor(bg_color)
        leg = ax.get_legend()
        if leg is not None:
            leg.get_frame().set_facecolor(bg_color)

    save_kwargs = {
        "bbox_inches": "tight",
        "facecolor": bg_color,
        "edgecolor": "none"
    }
    fig.savefig(pdf_path, **save_kwargs)
    fig.savefig(png_path, dpi=300, **save_kwargs)
    print(f"Saved: {pdf_path} and {png_path}")


def _style_axis(
    ax: plt.Axes,
    title: str,
    xlabel: str,
    ylabel: str,
    title_size: float = 12.5,
    legend_loc: str = "best",
    legend_size: float = 9.0,
    grid_axis: str = "both",
    grid_which: str = "major"
):
    ax.set_title(title, fontsize=title_size, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, axis=grid_axis, which=grid_which, alpha=0.3)
    ax.legend(fontsize=legend_size, loc=legend_loc)


def plot_training_comparison(
    flb_history: Optional[dict],
    baseline_histories: Dict[str, dict],
    title: str,
    filename: str,
    save_dir: str = "./figures",
    bg_color: str = "white",
    legend_loc: str = "lower right"
):
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    if flb_history is not None:
        ax.plot(
            flb_history["epochs"],
            flb_history["val_acc"],
            label=f"FLBViT ({max(flb_history['val_acc']):.2f}%)",
            color="black",
            linewidth=2.8,
            marker="o",
            markersize=4
        )

    for label, h in baseline_histories.items():
        ax.plot(
            h["epochs"],
            h["val_acc"],
            label=f"{label} ({max(h['val_acc']):.2f}%)",
            linewidth=2.0,
            linestyle="--"
        )

    _style_axis(
        ax, title=title, xlabel="Epoch", ylabel="Validation Accuracy (%)",
        title_size=13, legend_loc=legend_loc, legend_size=9.5
    )
    save_figure(fig, filename, save_dir, bg_color=bg_color)
    plt.show()


def plot_flb_losses(
    flb_history: dict,
    filename: str = "fig_flb_training_losses",
    save_dir: str = "./figures",
    bg_color: str = "white",
    legend_loc: str = "upper right"
):
    fig, ax1 = plt.subplots(figsize=(6.0, 4.5))
    ax1.plot(
        flb_history["epochs"], flb_history["train_ce"],
        label="Train Cross-Entropy Loss", color="tab:red", linewidth=2.4, marker="o", markersize=4
    )
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cross-Entropy Loss", color="tab:red", fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        flb_history["epochs"], flb_history["train_aux"],
        label="Intra-Layer JEPA Aux Loss", color="tab:purple", linewidth=2.4, linestyle="--", marker="s", markersize=4
    )
    ax2.set_ylabel("JEPA Auxiliary Cosine Loss", color="tab:purple", fontsize=11)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9.5, loc=legend_loc)
    ax1.set_title("FLBViT Joint Training Optimization", fontsize=13, fontweight="bold")
    save_figure(fig, filename, save_dir, bg_color=bg_color)
    plt.show()


def plot_noise_comparison(
    noise_results: dict,
    baseline_curves: Dict[str, List[float]],
    title: str,
    filename: str,
    save_dir: str = "./figures",
    bg_color: str = "white",
    legend_loc: str = "lower left"
):
    noise_levels = noise_results["noise_levels"]
    fig, ax = plt.subplots(figsize=(6.0, 4.5))

    ax.plot(
        noise_levels, noise_results["flb_step1_acc"],
        label="FLBViT (Step 1 - Feedforward)", color="dimgray", linestyle=":", linewidth=2.2, marker="v"
    )
    ax.plot(
        noise_levels, noise_results["flb_fixed12_acc"],
        label="FLBViT (Settled @ 12 Steps)", color="black", linewidth=2.8, marker="o"
    )

    for label, vals in baseline_curves.items():
        ax.plot(noise_levels, vals, label=label, linewidth=2.0, linestyle="--", marker="s", markersize=4)

    _style_axis(
        ax, title=title, xlabel="Gaussian Noise Std ($\\sigma$)", ylabel="Validation Accuracy (%)",
        legend_loc=legend_loc, legend_size=9.0
    )
    save_figure(fig, filename, save_dir, bg_color=bg_color)
    plt.show()


def plot_test_time_scaling(
    noise_results: dict,
    filename: str = "fig_test_time_scaling",
    save_dir: str = "./figures",
    bg_color: str = "white",
    legend_loc: str = "lower right"
):
    max_eval_steps = noise_results["max_eval_steps"]
    trained_limit = noise_results["trained_limit"]
    steps_x = np.arange(1, max_eval_steps + 1)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for sigma in noise_results["noise_levels"]:
        curve = noise_results["flb_step_curves"][sigma]
        gain_12 = curve[trained_limit - 1] - curve[0]
        ax.plot(steps_x, curve, linewidth=2.2, label=f"$\\sigma={sigma:.2f}$ (12-Step Gain: {gain_12:+.2f}%)")

    ax.axvline(trained_limit, color="black", linestyle="--", alpha=0.7, label=f"Trained Horizon ({trained_limit} Steps)")
    _style_axis(
        ax,
        title=f"FLBViT Test-Time Compute Scaling (Ticks 1 to {max_eval_steps})",
        xlabel="Recurrent Settling Tick",
        ylabel="Validation Accuracy (%)",
        legend_loc=legend_loc,
        legend_size=8.5
    )
    save_figure(fig, filename, save_dir, bg_color=bg_color)
    plt.show()


def plot_settling_gain(
    noise_results: dict,
    filename: str = "fig_settling_gain",
    save_dir: str = "./figures",
    bg_color: str = "white",
    legend_loc: str = "upper left"
):
    noise_levels = noise_results["noise_levels"]
    max_eval_steps = noise_results["max_eval_steps"]
    gain_12 = [s12 - s1 for s1, s12 in zip(noise_results["flb_step1_acc"], noise_results["flb_fixed12_acc"])]
    gain_50 = [s50 - s1 for s1, s50 in zip(noise_results["flb_step1_acc"], noise_results["flb_fixed50_acc"])]

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.plot(noise_levels, gain_12, label="Settling Boost @ 12 Steps (Acc12 - Acc1)", color="tab:green", linewidth=2.6, marker="o", markersize=6)
    ax.plot(noise_levels, gain_50, label=f"Settling Boost @ {max_eval_steps} Steps (Acc{max_eval_steps} - Acc1)", color="darkgreen", linestyle="--", linewidth=2.2, marker="*", markersize=7)
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=1.0)
    _style_axis(
        ax,
        title="FLBViT Recurrent Accuracy Gain Across Noise Levels",
        xlabel="Gaussian Noise Std ($\\sigma$)",
        ylabel="Accuracy Gain Over Step 1 (%)",
        legend_loc=legend_loc,
        legend_size=9.5
    )
    save_figure(fig, filename, save_dir, bg_color=bg_color)
    plt.show()


def plot_state_contraction(
    contraction_results: dict,
    filename: str = "fig_state_contraction",
    save_dir: str = "./figures",
    bg_color: str = "white",
    threshold: float = 0.01,
    legend_loc: str = "upper right"
):
    num_steps = contraction_results["num_steps"]
    n_run = contraction_results["num_batches_run"]
    max_mean = contraction_results["max_mean"]
    max_std = contraction_results["max_std"]
    steps_x = np.arange(2, num_steps + 1)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(steps_x, max_mean, label="Worst-Case Stream ($\\max \\Delta$)", color="black", linewidth=2.6)
    ax.fill_between(steps_x, np.maximum(max_mean - max_std, 1e-6), max_mean + max_std, color="black", alpha=0.12)

    ax.plot(steps_x, contraction_results["lat_mean"][0], label="Lateral L0 ($\\Delta h_0$)", color="tab:blue", linewidth=2.0)
    ax.plot(steps_x, contraction_results["lat_mean"][2], label="Lateral L2 ($\\Delta h_2$)", color="tab:cyan", linewidth=1.8, linestyle="--")
    ax.plot(steps_x, contraction_results["dec_mean"], label="Multiscale Decoder ($\\Delta g_{top}$)", color="tab:orange", linewidth=2.0)
    ax.plot(steps_x, contraction_results["back_mean"][2], label="Backward B2 ($\\Delta g_2$)", color="tab:red", linewidth=1.8, linestyle="-.")
    ax.plot(steps_x, contraction_results["fwd_mean"], label="Forward Output ($\\Delta x_2$)", color="tab:green", linewidth=1.8, linestyle=":")
    ax.axhline(threshold, color="crimson", linestyle="--", alpha=0.7, label=f"Halting Threshold ({threshold:g})")

    ax.set_yscale("log")
    _style_axis(
        ax,
        title=f"FLBViT Fixed-Point Attractor Contraction (Mean $\\pm$ Std, $N={n_run}$ Batches)",
        xlabel="Recurrent Settling Step",
        ylabel="Relative $L_2$ State Delta (Log Scale)",
        title_size=11.5,
        legend_loc=legend_loc,
        legend_size=8.5,
        grid_which="both"
    )
    save_figure(fig, filename, save_dir, bg_color=bg_color)
    plt.show()


def plot_reliability(
    reliability_results: dict,
    baseline_names: List[str],
    filename: str = "fig_reliability_calibration",
    save_dir: str = "./figures",
    bg_color: str = "white",
    legend_loc: str = "best"
):
    noise_levels = reliability_results["noise_levels"]
    reliability = reliability_results["reliability"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    flb_styles = [
        ("FLBViT (Step 1)", "dimgray", ":"),
        ("FLBViT (Step 12)", "black", "-"),
    ]

    for m_key, col, ls in flb_styles:
        if m_key in reliability[noise_levels[0]]:
            eces = [reliability[s][m_key]["ece"] for s in noise_levels]
            ents = [reliability[s][m_key]["mean_entropy"] for s in noise_levels]
            axes[0].plot(noise_levels, eces, label=m_key, color=col, linestyle=ls, linewidth=2.4, marker="o")
            axes[1].plot(noise_levels, ents, label=m_key, color=col, linestyle=ls, linewidth=2.4, marker="s")

    for m_key in baseline_names:
        if m_key in reliability[noise_levels[0]]:
            eces = [reliability[s][m_key]["ece"] for s in noise_levels]
            ents = [reliability[s][m_key]["mean_entropy"] for s in noise_levels]
            axes[0].plot(noise_levels, eces, label=m_key, linestyle="--", linewidth=2.0, marker="o")
            axes[1].plot(noise_levels, ents, label=m_key, linestyle="--", linewidth=2.0, marker="s")

    _style_axis(
        axes[0],
        title="Expected Calibration Error (ECE $\\downarrow$)",
        xlabel="Gaussian Noise Std ($\\sigma$)",
        ylabel="ECE (%) - Lower is Better",
        title_size=12.0,
        legend_loc=legend_loc,
        legend_size=8.5
    )
    _style_axis(
        axes[1],
        title="Predictive Entropy / Uncertainty ($H(y|x)$)",
        xlabel="Gaussian Noise Std ($\\sigma$)",
        ylabel="Mean Shannon Entropy (Nats)",
        title_size=12.0,
        legend_loc=legend_loc,
        legend_size=8.5
    )

    save_figure(fig, filename, save_dir, bg_color=bg_color)
    plt.show()


def plot_per_class_gain(
    reliability_results: dict,
    filename: str = "fig_per_class_gain",
    save_dir: str = "./figures",
    bg_color: str = "white",
    legend_loc: str = "best"
):
    noise_levels = reliability_results["noise_levels"]
    per_class_gains = reliability_results["per_class_gains"]

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    x_pos = np.arange(len(CIFAR10_CLASSES))
    width = 0.26
    offset_center = (len(noise_levels) - 1) / 2.0

    for idx, sigma in enumerate(noise_levels):
        ax.bar(
            x_pos + (idx - offset_center) * width,
            per_class_gains[sigma],
            width=width,
            label=f"$\\sigma={sigma:.2f}$"
        )

    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=25, ha="right", fontsize=10)
    _style_axis(
        ax,
        title="Class-Specific Accuracy Gain From Recurrent Settling",
        xlabel="",
        ylabel="Recurrent Accuracy Gain (Acc12 - Acc1, %)",
        legend_loc=legend_loc,
        legend_size=9.5,
        grid_axis="y"
    )
    save_figure(fig, filename, save_dir, bg_color=bg_color)
    plt.show()


def plot_qualitative_cases(
    cases: list,
    trained_limit: int,
    title_prefix: str,
    filename: str,
    save_dir: str = "./figures",
    bg_color: str = "white"
):
    n_rows = len(cases)
    if n_rows == 0:
        print(f"No cases to plot for {filename}.")
        return

    fig, axes = plt.subplots(n_rows, 3, figsize=(10.5, 3.1 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    plt.subplots_adjust(wspace=0.30, hspace=0.38)

    ticks = np.arange(1, trained_limit + 1)
    for r, (img_t, y_idx, p1_idx, p12_idx, s_probs, attn_8x8) in enumerate(cases):
        img_np = img_t.permute(1, 2, 0).numpy()

        axes[r, 0].imshow(img_np)
        axes[r, 0].set_title(
            f"True: {CIFAR10_CLASSES[y_idx]}\nStep 1: {CIFAR10_CLASSES[p1_idx]} -> Step 12: {CIFAR10_CLASSES[p12_idx]}",
            fontsize=9.5, fontweight="bold"
        )
        axes[r, 0].axis("off")

        attn_up = F.interpolate(
            attn_8x8.unsqueeze(0).unsqueeze(0), size=(32, 32), mode="bilinear", align_corners=False
        ).squeeze().numpy()
        axes[r, 1].imshow(img_np)
        axes[r, 1].imshow(attn_up, cmap="jet", alpha=0.45)
        axes[r, 1].set_title("Step 12 Decoder Attention Map", fontsize=9.5, fontweight="bold")
        axes[r, 1].axis("off")

        true_curve = [sp[y_idx].item() * 100.0 for sp in s_probs]
        p1_curve = [sp[p1_idx].item() * 100.0 for sp in s_probs]
        axes[r, 2].plot(ticks, true_curve, label=f"True ({CIFAR10_CLASSES[y_idx]})", color="tab:green", linewidth=2.4, marker="o", markersize=4)
        axes[r, 2].plot(ticks, p1_curve, label=f"Step 1 Pred ({CIFAR10_CLASSES[p1_idx]})", color="tab:red", linestyle="--", linewidth=2.0, marker="s", markersize=4)
        if p12_idx != y_idx and p12_idx != p1_idx:
            p12_curve = [sp[p12_idx].item() * 100.0 for sp in s_probs]
            axes[r, 2].plot(ticks, p12_curve, label=f"Step 12 Pred ({CIFAR10_CLASSES[p12_idx]})", color="tab:orange", linestyle="-.", linewidth=2.0)

        axes[r, 2].set_ylim(0, 105)
        _style_axis(
            axes[r, 2],
            title=f"{title_prefix}: Confidence Trajectory",
            xlabel="Recurrent Tick",
            ylabel="Confidence (%)",
            title_size=9.5,
            legend_size=8.0
        )

    save_figure(fig, filename, save_dir, bg_color=bg_color)
    plt.show()