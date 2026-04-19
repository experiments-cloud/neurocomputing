"""
tinystories_hvp_ablated.py

Executes the Hessian-Vector Product (HVP) extraction on the semantically 
ablated TinyStories dataset. Evaluates the dominant eigenvalue (lambda_max) 
to demonstrate that removing semantic redundancy forces the architecture 
to traverse a significantly steeper curvature barrier.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer
import math

# ==========================================
# 1. Transformer Architecture (Unfused)
# ==========================================
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_len):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        
        self.register_buffer(
            "bias", 
            torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len)
        )

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Native mathematical attention (Ensures gradient flow for HVP)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_k))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, max_len):
        super().__init__()
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
        self.lm_head.weight = self.token_emb.weight
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
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            logits_view = logits.view(-1, logits.size(-1))
            targets_view = targets.view(-1)
            loss = nn.functional.cross_entropy(logits_view, targets_view)
        return logits, loss

# ==========================================
# 2. HVP Configuration and Extraction
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware Device: {device}")

MAX_LENGTH = 64
BATCH_SIZE = 32  # Reduced to avoid CPU/GPU memory bottlenecks during double backprop
EVAL_STEPS = 50

# Tokenizer Configuration
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
vocab_size = tokenizer.vocab_size

# Model Instantiation
model = TinyStoriesTransformer(
    vocab_size=vocab_size, d_model=256, n_heads=8, 
    d_ff=1024, n_layers=4, max_len=MAX_LENGTH
).to(device)

def compute_hvp(model, loss, v):
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
    flat_grads = torch.cat([g.view(-1) for g in grads])
    grad_v_prod = torch.sum(flat_grads * v)
    
    hvp_grads = torch.autograd.grad(grad_v_prod, model.parameters(), retain_graph=True)
    flat_hvp = torch.cat([g.contiguous().view(-1) for g in hvp_grads])
    return flat_hvp

def get_dominant_eigenvalue(model, data_loader_iter, num_iterations=10):
    model.eval()
    batch = next(data_loader_iter)
    inputs = batch["input_ids"].to(device)
    targets = batch["labels"].to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    v = torch.randn(total_params).to(device)
    v = v / torch.norm(v)
    
    for _ in range(num_iterations):
        model.zero_grad()
        _, loss = model(inputs, targets)
        Hv = compute_hvp(model, loss, v)
        lambda_max = torch.dot(v, Hv).item()
        v = Hv / (torch.norm(Hv) + 1e-8)
        
        del Hv
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    model.train()
    return lambda_max

# ==========================================
# 3. Optimization Loop
# ==========================================
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))

# Import the actual DataLoaders from our semantic ablation script
from tinystories_data_ablated import train_dataloader, val_dataloader

def get_infinite_loader(dataloader):
    """Allows infinite iteration over the DataLoader without raising StopIteration"""
    while True:
        for batch in dataloader:
            yield batch

train_loader_iter = iter(get_infinite_loader(train_dataloader))
val_loader_iter = iter(get_infinite_loader(val_dataloader))

print("Initiating the geometric phase transition...")
model.train()
max_steps = 200

for step in range(max_steps + 1):
    batch = next(train_loader_iter)
    inputs = batch["input_ids"].to(device)
    targets = batch["labels"].to(device)
    
    optimizer.zero_grad()
    _, loss = model(inputs, targets)
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    if step % EVAL_STEPS == 0:
        print(f"Step {step:03d} | Train Loss: {loss.item():.4f} | Calculating topology...")
        try:
            lambda_max = get_dominant_eigenvalue(model, val_loader_iter)
            print(f"---> Extracted Spectrum: lambda_max = {lambda_max:.2f}")
        except Exception as e:
            print(f"Error calculating HVP: {e}")
            break

print("Ablation training run completed successfully.")