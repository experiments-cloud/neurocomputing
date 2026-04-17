"""
generate_dataset.py

Generates the foundational synthetic dataset for grokking experiments.
Task: Modular Addition a + b (mod p).

This module isolates the grokking phase transition by forcing the neural network
to learn a cyclical mathematical algorithm (modulo arithmetic) rather than 
memorizing heuristic pairs. It is designed to evaluate both standard generalization
and the structural limits of learning via data sparsity ablation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ModularArithmeticDataset(Dataset):
    """
    Dataset generator for the modular arithmetic task: a + b (mod p).
    """
    
    def __init__(self, p: int = 97, split: str = 'train', train_ratio: float = 0.5, seed: int = 42):
        """
        Initializes the dataset and performs deterministic orthogonal splits.

        =====================================================================
        🔬 ABLATION STUDY CONFIGURATION (Ref: Section 5.3 of the paper)
        =====================================================================
        Modify the 'train_ratio' parameter to reproduce the structural 
        limits of generalization:
        
        * train_ratio = 0.50 (Base) -> Relative abundance. Fast generalization.
        * train_ratio = 0.25 (Grok) -> Moderate sparsity. Induces standard grokking 
                                       (delayed algorithmic generalization).
        * train_ratio = 0.10 (Fail) -> Severe sparsity. Model remains confined to
                                       a high-curvature sharp minimum (permanent overfitting).
        =====================================================================

        Args:
            p (int): Prime number defining the modulo group (Z_p). Default is 97.
            split (str): 'train' for training set, 'val' for validation set.
            train_ratio (float): Fraction of the total data used for training. 
            seed (int): Random seed for strict reproducibility across experiments.
        """
        self.p = p
        self.split = split
        self.vocab_size = p + 3
        
        # Special algorithmic tokens
        self.OP_TOKEN = p       # Represents the '+' operator
        self.EQ_TOKEN = p + 1   # Represents the '=' operator
        self.EOS_TOKEN = p + 2  # Represents End-Of-Sequence (optional padding)
        
        # Generate the full universe of (a, b) combinations (Total: p^2)
        a_vals = torch.arange(p)
        b_vals = torch.arange(p)
        grid_a, grid_b = torch.meshgrid(a_vals, b_vals, indexing='ij')
        
        self.a_data = grid_a.flatten()
        self.b_data = grid_b.flatten()
        
        # Calculate ground truth: c = (a + b) % p
        self.c_data = (self.a_data + self.b_data) % self.p
        
        # Deterministic Train / Validation split
        total_samples = len(self.a_data)
        indices = np.arange(total_samples)
        
        # Ensures orthogonal split: the model must generalize to unseen combinations
        np.random.seed(seed)
        np.random.shuffle(indices)
        
        train_size = int(total_samples * train_ratio)
        
        if split == 'train':
            self.indices = indices[:train_size]
        else:
            self.indices = indices[train_size:]
            
        print(f"Dataset '{split}' successfully initialized with {len(self.indices)} samples "
              f"(Ratio: {train_ratio * 100}%). Modulo p={self.p}.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        a = self.a_data[real_idx]
        b = self.b_data[real_idx]
        c = self.c_data[real_idx]
        
        # Autoregressive setup (Teacher Forcing)
        # Sequence format: [a, +, b, =]
        # Target format:   [+, b, =, c] (Shifted right by 1 for next-token prediction)
        x = torch.tensor([a, self.OP_TOKEN, b, self.EQ_TOKEN], dtype=torch.long)
        y = torch.tensor([self.OP_TOKEN, b, self.EQ_TOKEN, c], dtype=torch.long)
        
        return x, y

# --- Execution and Sanity Check ---
if __name__ == "__main__":
    # Example execution demonstrating the 25% sparsity setup
    print("--- Executing Sanity Check ---")
    train_dataset = ModularArithmeticDataset(p=97, split='train', train_ratio=0.25)
    val_dataset = ModularArithmeticDataset(p=97, split='val', train_ratio=0.25)
    
    # Batch size of 256 is optimal for small models and fast iteration
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    x_batch, y_batch = next(iter(train_loader))
    
    print("\n--- Dataset Telemetry ---")
    print(f"Input tensor X shape: {x_batch.shape} -> (Batch_size, Sequence_length)")
    print(f"Target tensor Y shape: {y_batch.shape}")
    print(f"\nDecoded example of the first sequence in batch:")
    print(f"Input X: [a={x_batch[0][0]}, op={x_batch[0][1]}, b={x_batch[0][2]}, eq={x_batch[0][3]}]")
    print(f"Target Y (to predict): {y_batch[0]}")