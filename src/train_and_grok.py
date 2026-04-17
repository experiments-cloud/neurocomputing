"""
train_and_grok.py

Executes the long-horizon training loop to induce the grokking phase transition.
This script natively supports ablation studies by adjusting the weight decay parameter.
It automatically saves model checkpoints and telemetry (loss, accuracy) necessary 
for the post-hoc Hessian topology analysis.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import our custom modules
from generate_dataset import ModularArithmeticDataset
from model_architecture import ToyTransformer

# =====================================================================
# 🔬 HYPERPARAMETER CONFIGURATION & ABLATION PANEL
# =====================================================================
P_MODULO = 97
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
MAX_STEPS = 25000   # Optimization steps (essential for asymptotic regime)
EVAL_EVERY = 100    # Evaluate and save telemetry every N steps
SEED = 42

# --- ABLATION STUDY: WEIGHT DECAY ---
# Modify this parameter to replicate the grokking induction or its ablation:
# 1.0 -> Catalyst for grokking (Forces topological compression)
# 0.0 -> Ablation (Model memorizes but never transitions to generalization)
WEIGHT_DECAY = 1.0  
# =====================================================================


def train_model() -> None:
    """
    Main training loop. Initializes the dataset, model, and optimizer,
    and executes the optimization over MAX_STEPS while recording telemetry.
    """
    # 1. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing training on device: {device}")
    print(f"Configuration -> Weight Decay: {WEIGHT_DECAY} | Max Steps: {MAX_STEPS}")

    # 2. Initialize Data
    # Utilizing drop_last=True ensures consistent batch sizes for stable Hessian computation later
    train_dataset = ModularArithmeticDataset(p=P_MODULO, split='train', seed=SEED)
    val_dataset = ModularArithmeticDataset(p=P_MODULO, split='val', seed=SEED)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    # 3. Initialize Model & Optimizer
    model = ToyTransformer(vocab_size=train_dataset.vocab_size).to(device)
    
    # Using AdamW to decouple weight decay from momentum-based gradient updates
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY, 
        betas=(0.9, 0.98)
    )
    criterion = nn.CrossEntropyLoss()

    # 4. Telemetry and Checkpoint tracking setup
    telemetry = {
        "steps": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": []
    }
    os.makedirs("checkpoints", exist_ok=True)

    # 5. Training Loop
    model.train()
    step = 0
    train_iterator = iter(train_loader)

    print("\nStarting the optimization loop...")
    print("This may take a while depending on your hardware.")

    while step < MAX_STEPS:
        try:
            x_batch, y_batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            x_batch, y_batch = next(train_iterator)
            
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(x_batch)
        
        # We only care about predicting the LAST token (the actual result 'c')
        # logits shape: (batch_size, seq_len, vocab_size)
        final_logits = logits[:, -1, :]
        final_targets = y_batch[:, -1]
        
        loss = criterion(final_logits, final_targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # --- Evaluation and Telemetry Logging ---
        if step % EVAL_EVERY == 0:
            model.eval()
            val_loss_total = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    v_logits = model(x_val)
                    v_final_logits = v_logits[:, -1, :]
                    v_final_targets = y_val[:, -1]
                    
                    v_loss = criterion(v_final_logits, v_final_targets)
                    val_loss_total += v_loss.item()
                    
                    # Calculate Accuracy
                    predictions = torch.argmax(v_final_logits, dim=-1)
                    correct_predictions += (predictions == v_final_targets).sum().item()
                    total_predictions += v_final_targets.size(0)
                    
            avg_val_loss = val_loss_total / len(val_loader)
            val_acc = correct_predictions / total_predictions
            
            # Save telemetry
            telemetry["steps"].append(step)
            telemetry["train_loss"].append(loss.item())
            telemetry["val_loss"].append(avg_val_loss)
            telemetry["val_accuracy"].append(val_acc)
            
            print(f"Step {step:05d} | Train Loss: {loss.item():.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Save model weights (Checkpointing)
            # High-resolution checkpointing at the beginning, spaced out later to save disk space
            if step < 2000 or step % (EVAL_EVERY * 5) == 0:
                torch.save(model.state_dict(), f"checkpoints/model_step_{step}.pt")
                
            model.train()  # Return to training mode
            
        step += 1

    # 6. Save final telemetry to disk
    telemetry_file = "grokking_telemetry.json"
    with open(telemetry_file, "w") as f:
        json.dump(telemetry, f)

    print(f"\nTraining complete! Telemetry saved to '{telemetry_file}'.")


if __name__ == "__main__":
    train_model()