"""
model_architecture.py

Defines a minimalistic causal Transformer (Decoder-only) optimized 
for grokking experiments and Hessian eigenvalue analysis.

The architectural specifications (d_model, n_heads, n_layers) perfectly match
Table 1 of the paper. This compact size allows for the iterative resolution 
of the Hessian matrix without encountering out-of-memory (OOM) errors.
"""

import torch
import torch.nn as nn


class ToyTransformer(nn.Module):
    """
    A lightweight, GPT-style autoregressive Transformer.
    Dimensions are kept small (d_model=128, 2 layers) to allow for 
    computationally feasible Hessian extraction during training.
    """
    
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int = 128, 
                 n_heads: int = 4, 
                 n_layers: int = 2, 
                 max_seq_len: int = 4, 
                 dropout: float = 0.1):
        """
        Initializes the Transformer architecture.
        
        Args:
            vocab_size (int): Size of the vocabulary (usually p + 3).
            d_model (int): Dimensionality of the embeddings and hidden states (Table 1).
            n_heads (int): Number of attention heads (Table 1).
            n_layers (int): Number of Transformer blocks (Table 1).
            max_seq_len (int): Maximum sequence length (4 for [a, +, b, =]).
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.d_model = d_model
        
        # 1. Embeddings: Tokens and Absolute Positions
        # Since seq_len is extremely small (4), standard learned positional embeddings are optimal.
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        # 2. Transformer Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 4, 
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN architecture (Standard in modern LLMs)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 3. Output Projection (Language Modeling Head)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # 4. Weight Initialization
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Xavier uniform initialization helps stabilize the initial 
        loss landscape topology, providing a clean starting point.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Generates a causal mask (upper triangular matrix) to prevent 
        attention to future tokens. Essential for autoregressive learning.
        
        Args:
            sz (int): Sequence length.
            
        Returns:
            torch.Tensor: The causal mask.
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the causal Transformer.
        
        Args:
            x (torch.Tensor): Input sequence of shape (batch_size, seq_len).
            
        Returns:
            torch.Tensor: Logits of shape (batch_size, seq_len, vocab_size).
        """
        batch_size, seq_len = x.size()
        
        # Create positional indices dynamically based on input sequence length
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)
        
        # Combine embeddings
        x_emb = self.token_emb(x) + self.pos_emb(positions)
        
        # Generate and apply causal mask
        causal_mask = self.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Pass through Transformer blocks
        # PyTorch uses 'mask' for sequence causality in Encoder layers used causally
        out = self.transformer(x_emb, mask=causal_mask, is_causal=True)
        
        # Map to vocabulary dimension
        logits = self.lm_head(out)
        
        return logits


# --- Sanity Check ---
if __name__ == "__main__":
    # Test parameters (matching Phase 1: p=97 -> vocab_size=100)
    vocab_size = 100
    batch_size = 256
    seq_len = 4
    
    # Initialize Model
    model = ToyTransformer(vocab_size=vocab_size)
    
    # Create a dummy input tensor matching DataLoader output shape
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    logits = model(dummy_input)
    
    # Parameter count calculation (Should match ~422,000 from Table 1)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n--- Model Architecture Telemetry ---")
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output logits shape: {logits.shape} -> (Batch, Seq_Len, Vocab_Size)")