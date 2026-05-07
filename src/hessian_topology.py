"""
hessian_topology.py

Calculates the dominant eigenvalue (lambda_max) of the Hessian matrix 
for saved optimization states to mathematically prove geometric compression 
during delayed algorithmic generalization. Utilizes Power Iteration and 
Hessian-Vector Products (HVP) to strictly bound spatial complexity to O(N), 
bypassing intractable O(N^2) or O(N^3) memory constraints.
"""

import os
import glob
import json
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import custom modules
from generate_dataset import ModularArithmeticDataset
from model_architecture import ToyTransformer

# =====================================================================
# ⚙️ CONFIGURATION
# =====================================================================
P_MODULO = 97
BATCH_SIZE = 512            # Larger batch size to ensure stability in empirical Hessian approximation
NUM_POWER_ITERATIONS = 20   # Asymptotic iterations to guarantee convergence to the dominant eigenvector
SEED = 42
# =====================================================================


def compute_hvp(model: nn.Module, loss: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Computes the exact Hessian-Vector Product utilizing automatic differentiation.
    
    Args:
        model (nn.Module): The autoregressive neural network architecture.
        loss (torch.Tensor): The scalar loss value (must be calculated with create_graph=True).
        v (torch.Tensor): The arbitrary vector to multiply against the Hessian operator.
        
    Returns:
        torch.Tensor: The resulting directional derivative vector from the H * v product.
    """
    # 1st derivative: Gradient of the objective function with respect to parameters
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
    
    # Flatten parametric gradients into a single 1D vector
    grad_vector = torch.cat([g.contiguous().view(-1) for g in grads])
    
    # Dot product of the gradient vector and the random vector 'v'
    grad_v_prod = torch.sum(grad_vector * v)
    
    # 2nd derivative: Gradient of the dot product yields the exact HVP
    hvp_grads = torch.autograd.grad(grad_v_prod, model.parameters(), retain_graph=True)
    hvp_vector = torch.cat([g.contiguous().view(-1) for g in hvp_grads])
    
    return hvp_vector


def get_dominant_eigenvalue(model: nn.Module, 
                            inputs: torch.Tensor, 
                            targets: torch.Tensor, 
                            criterion: nn.Module, 
                            device: torch.device,
                            num_iterations: int = 20) -> Tuple[float, float]:
    """
    Approximates the lambda_max of the Hessian matrix utilizing the Power Iteration method.
    Integrates a critical compiler-level intervention to mathematically enable 
    exact double backpropagation flow.
    """
    model.eval()  # Bound variance constraints during active spectral evaluation
    
    # --- CRITICAL FIX FOR EXACT DOUBLE BACKPROP (HESSIAN EXTRACTION) IN PYTORCH ---
    # Hardware-level fused attention kernels (FlashAttention/MemEfficient) operate as opaque
    # abstractions lacking support for exact second-order functions.
    # We explicitly force the compiler to unroll the full computational graph using native Math.
    try:
        # PyTorch >= 2.1.2
        from torch.nn.attention import sdpa_kernel, SDPBackend
        sdp_context = sdpa_kernel(SDPBackend.MATH)
    except ImportError:
        # PyTorch 2.0 - 2.1.1 fallback
        sdp_context = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
        
    with sdp_context:
        # Forward pass (Computational graph strictly utilizes native tensor operations)
        logits = model(inputs)
        final_logits = logits[:, -1, :]
        final_targets = targets[:, -1]
        loss = criterion(final_logits, final_targets)
    
    # Initialize a random vector 'v' matching the dimensionality of the parameter space
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    v = torch.randn(total_params, device=device)
    v = v / torch.norm(v)  # L2 Normalization to unit length
    
    lambda_max = 0.0
    
    for _ in range(num_iterations):
        # Compute H*v (Mathematical flow guaranteed by the unfused graph)
        Hv = compute_hvp(model, loss, v)
        
        # Rayleigh quotient approximation of maximum directional curvature: v^T * H * v
        lambda_max = torch.dot(v, Hv).item()
        
        # Asymptotic normalization to actively prevent arithmetic overflow
        v = Hv / (torch.norm(Hv) + 1e-8)
        
    return lambda_max, loss.item()


def analyze_checkpoints() -> None:
    """
    Iterates over saved optimization states, continuously extracts the dominant eigenvalue 
    for each, and augments the macroscopic telemetry for final continuous visualization.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing continuous spectral analysis on hardware architecture: {device}")

    # 1. Load a fixed static batch to represent the optimization manifold consistently
    train_dataset = ModularArithmeticDataset(p=P_MODULO, split='train', seed=SEED)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    x_batch, y_batch = next(iter(train_loader))
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    
    criterion = nn.CrossEntropyLoss()

    # 2. Locate and chronologically sort saved parametric states
    checkpoint_files = glob.glob("checkpoints/model_step_*.pt")
    checkpoint_files.sort(key=lambda x: int(x.split('_step_')[1].split('.pt')[0]))
    
    if not checkpoint_files:
        print("Error: No saved states found in 'checkpoints/' directory.")
        return

    # 3. Load existing macroscopic optimization trajectory
    telemetry_path = "grokking_telemetry.json"
    try:
        with open(telemetry_path, "r") as f:
            telemetry = json.load(f)
    except FileNotFoundError:
        print(f"Error: '{telemetry_path}' not found. Execute optimization trajectory first.")
        return

    # Augment JSON with mathematical arrays for spectral tracking
    telemetry["lambda_max"] = []
    telemetry["checkpoint_steps"] = []

    print(f"\nDiscovered {len(checkpoint_files)} optimization states. Initiating Spectral Dissection...")
    
    # Initialize baseline architectural structure
    model = ToyTransformer(vocab_size=train_dataset.vocab_size).to(device)

    # 4. Extract continuous topology iteratively
    for ckpt_path in checkpoint_files:
        step = int(ckpt_path.split('_step_')[1].split('.pt')[0])
        
        # Restore geometric parameter state
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        
        # Calculate maximum directional curvature (lambda_max)
        print(f"Extracting local topology for Step {step}...", end=" ", flush=True)
        eig_val, _ = get_dominant_eigenvalue(model, x_batch, y_batch, criterion, device, NUM_POWER_ITERATIONS)
        print(f"lambda_max: {eig_val:.4f}")
        
        telemetry["checkpoint_steps"].append(step)
        telemetry["lambda_max"].append(eig_val)

    # 5. Persist augmented continuous telemetry
    output_path = "grokking_telemetry_with_hessian.json"
    with open(output_path, "w") as f:
        json.dump(telemetry, f)
        
    print(f"\nPhase 4 Complete! Continuous spectral trajectory successfully saved to '{output_path}'.")


if __name__ == "__main__":
    analyze_checkpoints()
