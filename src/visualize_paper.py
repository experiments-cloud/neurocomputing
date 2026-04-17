"""
visualize_paper.py

Generates camera-ready academic plots for the Grokking and Hessian topology paper.
Reads the augmented JSON telemetry and outputs a high-resolution PNG containing 
the 3-panel macroscopic and spectral dynamics.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns

def setup_academic_style() -> None:
    """
    Configures matplotlib and seaborn for publication-quality figures.
    """
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update({
        "font.family": "serif",
        "figure.autolayout": True,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "lines.linewidth": 2,
        "lines.markersize": 6,
    })

def generate_paper_figure(telemetry_file: str = "grokking_telemetry_with_hessian.json", 
                          output_filename: str = "figure_1_grokking_hessian.png") -> None:
    """
    Reads the telemetry data and generates the 3-panel figure demonstrating 
    the phase transition of grokking and the corresponding Hessian topology.
    
    Args:
        telemetry_file (str): Path to the augmented telemetry JSON.
        output_filename (str): Name of the output image file.
    """
    try:
        with open(telemetry_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find '{telemetry_file}'. Ensure the Hessian analysis completed.")
        return

    # Extract standard telemetry
    steps = data["steps"]
    train_loss = data["train_loss"]
    val_loss = data["val_loss"]
    val_acc = data["val_accuracy"]

    # Extract Hessian telemetry
    hessian_steps = data["checkpoint_steps"]
    lambda_max = data["lambda_max"]

    # Apply styling
    setup_academic_style()

    # Create a 3-panel figure sharing the X-axis
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # --- Panel 1: Loss Landscape ---
    ax1.plot(steps, train_loss, label="Train Loss", color="#1f77b4", alpha=0.8)
    ax1.plot(steps, val_loss, label="Validation Loss", color="#ff7f0e", alpha=0.8)
    ax1.set_yscale("log")
    ax1.set_ylabel("Cross-Entropy Loss (Log)")
    ax1.set_title("A. Training and Validation Loss Trajectories")
    ax1.legend(loc="upper right")
    
    # Highlight the grokking transition zone
    ax1.axvspan(300, 2500, color='gray', alpha=0.1)

    # --- Panel 2: Generalization (Accuracy) ---
    ax2.plot(steps, val_acc, label="Validation Accuracy", color="#2ca02c")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("B. Phase Transition in Generalization (Grokking)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="lower right")
    ax2.axvspan(300, 2500, color='gray', alpha=0.1)

    # --- Panel 3: Hessian Topology ---
    # We use markers here because Hessian is computed at discrete checkpoints
    ax3.plot(hessian_steps, lambda_max, marker='o', linestyle='-', color="#d62728", 
             markersize=4, label=r"Dominant Eigenvalue ($\lambda_{max}$)")
    ax3.set_ylabel(r"$\lambda_{max}$ of $\nabla^2 \mathcal{L}(\theta)$")
    ax3.set_xlabel("Optimization Steps (AdamW)")
    ax3.set_title("C. Spectral Dynamics of the Loss Hessian")
    ax3.legend(loc="upper right")
    ax3.axvspan(300, 2500, color='gray', alpha=0.1)

    # Polish and save
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"\nSuccess! Camera-ready figure saved as '{output_filename}'.")

if __name__ == "__main__":
    generate_paper_figure()