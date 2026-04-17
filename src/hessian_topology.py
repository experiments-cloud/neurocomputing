"""
hessian_topology.py

Calculates the dominant eigenvalue (lambda_max) of the Hessian matrix 
for saved model checkpoints to prove topological compression during grokking.
Utilizes Power Iteration and Hessian-Vector Products (HVP) to strictly bound 
spatial complexity to O(N), avoiding intractable O(N^2) or O(N^3) memory constraints.
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
BATCH_SIZE = 512            # Larger batch for more stable empirical Hessian estimation
NUM_POWER_ITERATIONS = 20   # Iterations to converge to the dominant eigenvector
SEED = 42
# =====================================================================


def compute_hvp(model: nn.Module, loss: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Computes the Hessian-Vector Product exactly using PyTorch's autograd.
    
    Args:
        model (nn.Module): The neural network architecture.
        loss (torch.Tensor): The scalar loss value (must be calculated with create_graph=True).
        v (torch.Tensor): The vector to multiply with the Hessian.
        
    Returns:
        torch.Tensor: The resulting vector from the H * v product.
    """
    # 1st derivative: Gradient of the loss with respect to parameters
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
    
    # Flatten gradients into a single 1D vector
    grad_vector = torch.cat([g.contiguous().view(-1) for g in grads])
    
    # Dot product of the gradient vector and our random vector 'v'
    grad_v_prod = torch.sum(grad_vector * v)
    
    # 2nd derivative: Gradient of the dot product yields the HVP
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
    Finds lambda_max of the Hessian using the Power Iteration method.
    Includes a critical compiler-level intervention to allow Double Backpropagation.
    """
    model.eval()  # Freeze dropout/batchnorm for stable evaluation
    
    # --- CRITICAL FIX FOR DOUBLE BACKPROP (HESSIAN) IN PYTORCH 2.X ---
    # Fused attention kernels (FlashAttention/MemEfficient) are opaque to second derivatives.
    # We explicitly force PyTorch to unroll the full computational graph using native Math.
    try:
        # PyTorch >= 2.1.2
        from torch.nn.attention import sdpa_kernel, SDPBackend
        sdp_context = sdpa_kernel(SDPBackend.MATH)
    except ImportError:
        # PyTorch 2.0 - 2.1.1 fallback
        sdp_context = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
        
    with sdp_context:
        # Forward pass (Graph is built using standard math operations supporting autograd)
        logits = model(inputs)
        final_logits = logits[:, -1, :]
        final_targets = targets[:, -1]
        loss = criterion(final_logits, final_targets)
    
    # Initialize a random vector 'v' of the same size as total parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    v = torch.randn(total_params, device=device)
    v = v / torch.norm(v)  # Normalize to unit length
    
    lambda_max = 0.0
    
    for _ in range(num_iterations):
        # Compute H*v (Succeeds because the graph supports double backprop)
        Hv = compute_hvp(model, loss, v)
        
        # Rayleigh quotient approximation of the maximum eigenvalue: v^T * H * v
        lambda_max = torch.dot(v, Hv).item()
        
        # Update and normalize v for the next iteration (prevents overflow)
        v = Hv / (torch.norm(Hv) + 1e-8)
        
    return lambda_max, loss.item()


def analyze_checkpoints() -> None:
    """
    Iterates over saved training checkpoints, calculates the dominant eigenvalue 
    for each, and augments the telemetry JSON for final visualization.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Hessian analysis on device: {device}")

    # 1. Load a fixed batch of data to represent the empirical risk surface consistently
    train_dataset = ModularArithmeticDataset(p=P_MODULO, split='train', seed=SEED)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    x_batch, y_batch = next(iter(train_loader))
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    
    criterion = nn.CrossEntropyLoss()

    # 2. Find and sort checkpoint files
    checkpoint_files = glob.glob("checkpoints/model_step_*.pt")
    checkpoint_files.sort(key=lambda x: int(x.split('_step_')[1].split('.pt')[0]))
    
    if not checkpoint_files:
        print("Error: No checkpoints found in 'checkpoints/' directory.")
        return

    # 3. Load existing macroscopic telemetry
    telemetry_path = "grokking_telemetry.json"
    try:
        with open(telemetry_path, "r") as f:
            telemetry = json.load(f)
    except FileNotFoundError:
        print(f"Error: '{telemetry_path}' not found. Run training script first.")
        return

    # Append new arrays for mathematical metrics
    telemetry["lambda_max"] = []
    telemetry["checkpoint_steps"] = []

    print(f"\nFound {len(checkpoint_files)} checkpoints. Starting Spectral Analysis...")
    
    # Initialize a fresh model structure
    model = ToyTransformer(vocab_size=train_dataset.vocab_size).to(device)

    # 4. Extract topology iteratively
    for ckpt_path in checkpoint_files:
        step = int(ckpt_path.split('_step_')[1].split('.pt')[0])
        
        # Load weights
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        
        # Calculate lambda_max
        print(f"Extracting Hessian topology for Step {step}...", end=" ", flush=True)
        eig_val, _ = get_dominant_eigenvalue(model, x_batch, y_batch, criterion, device, NUM_POWER_ITERATIONS)
        print(f"lambda_max: {eig_val:.4f}")
        
        telemetry["checkpoint_steps"].append(step)
        telemetry["lambda_max"].append(eig_val)

    # 5. Save the augmented telemetry
    output_path = "grokking_telemetry_with_hessian.json"
    with open(output_path, "w") as f:
        json.dump(telemetry, f)
        
    print(f"\nPhase 4 Complete! Augmented telemetry saved to '{output_path}'.")


if __name__ == "__main__":
    analyze_checkpoints()