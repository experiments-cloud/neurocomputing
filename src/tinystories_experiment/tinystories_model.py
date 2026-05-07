"""
tinystories_model.py

Defines the 16M parameter causal Transformer for the TinyStories dataset.
Crucially implements native Math-based attention (unfused) to ensure the 
computational graph allows for the double backward pass required to extract 
the Hessian dominant eigenvalue via HVP.
"""

import torch
import torch.nn as nn
import math

# ==========================================
# Mathematical Attention Module (Unfused Computational Graph)
# ==========================================
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_len):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        
        # Explicit upper triangular causal mask registered as a buffer
        # Mathematically prevents acausal information leakage
        self.register_buffer(
            "bias", 
            torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len)
        )

    def forward(self, x):
        B, T, C = x.size()
        
        # Derivation of Query, Key, and Value tensors
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Explicit native attention computation 
        # (Guarantees exact double backpropagation flow for HVP extraction)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_k))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.c_proj(y)

# ==========================================
# Transformer Block (Pre-LN Configuration)
# ==========================================
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, max_len):
        super().__init__()
        # Pre-LN configuration explicitly bounds the variance of hidden states
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_len)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# ==========================================
# Primary Architecture: TinyStoriesTransformer
# ==========================================
class TinyStoriesTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, d_ff=1024, n_layers=4, max_len=64):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        
        self.blocks = nn.Sequential(
            *[TransformerBlock(d_model, n_heads, d_ff, max_len) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying: Coupling embedding and LM head weights to optimize spatial complexity
        self.lm_head.weight = self.token_emb.weight
        
        # Topological Conditioning: Strict Xavier parameter initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        
        # Initial latent manifold representation
        x = self.token_emb(idx) + self.pos_emb(pos)
        
        # Forward propagation through structural blocks
        x = self.blocks(x)
        x = self.ln_f(x)
        
        # Linear projection into the discrete vocabulary space
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            # Flattening of logits and targets for precise Cross-Entropy evaluation
            logits_view = logits.view(-1, logits.size(-1))
            targets_view = targets.view(-1)
            loss = nn.functional.cross_entropy(logits_view, targets_view)
            
        return logits, loss

# ==========================================
# Instantiation and Independent Analytical Verification
# ==========================================
if __name__ == "__main__":
    from transformers import AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware architecture: {device}")

    # 1. Initialize tokenizer strictly to extract the vocabulary dimension (vocab_size)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    MAX_LENGTH = 64 # Ensure strict bounding of the spatial context

    # 2. Architecture instantiation
    model = TinyStoriesTransformer(
        vocab_size=vocab_size,
        d_model=256,
        n_heads=8,
        d_ff=1024,
        n_layers=4,
        max_len=MAX_LENGTH
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters (N): {total_params:,}")

    # 3. Generate a synthetic batch to mathematically verify graph compilation and forward propagation
    # Tensor dimensions: [Batch_Size=128, Sequence_Length=64]
    input_batch = torch.randint(0, vocab_size, (128, MAX_LENGTH)).to(device)
    labels_batch = torch.randint(0, vocab_size, (128, MAX_LENGTH)).to(device)

    logits, loss = model(input_batch, targets=labels_batch)
    print(f"Output logits tensor shape: {logits.shape}")
    print(f"Initial untrained cross-entropy loss: {loss.item():.4f}")
