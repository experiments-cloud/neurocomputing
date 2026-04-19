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
# Módulo de Atención Matemática (Sin Fused Kernels)
# ==========================================
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_len):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Proyecciones lineales
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        
        # Máscara causal (triangular superior) registrada como buffer
        self.register_buffer(
            "bias", 
            torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len)
        )

    def forward(self, x):
        B, T, C = x.size()
        
        # Cálculo de tensores Query, Key, Value
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Atención nativa explícita (Garantiza el flujo para HVP)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_k))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.c_proj(y)

# ==========================================
# Bloque Transformer (Pre-LN)
# ==========================================
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

# ==========================================
# Arquitectura Principal: TinyStoriesTransformer
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
        
        # Tie weights (Vincular pesos de embedding y salida para ahorrar RAM)
        self.lm_head.weight = self.token_emb.weight
        
        # Condicionamiento Topológico: Inicialización estricta Xavier
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
        
        # Representación latente inicial
        x = self.token_emb(idx) + self.pos_emb(pos)
        
        # Propagación a través de los bloques
        x = self.blocks(x)
        x = self.ln_f(x)
        
        # Proyección al vocabulario
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            # Flatten de logits y targets para Cross Entropy
            logits_view = logits.view(-1, logits.size(-1))
            targets_view = targets.view(-1)
            loss = nn.functional.cross_entropy(logits_view, targets_view)
            
        return logits, loss

# ==========================================
# Instanciación y Verificación Independiente
# ==========================================
from transformers import AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo de entrenamiento: {device}")

# 1. Cargamos rápidamente el tokenizador solo para obtener el vocab_size
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
vocab_size = tokenizer.vocab_size
MAX_LENGTH = 64 # Aseguramos que la constante esté definida

# 2. Instanciamos el modelo
model = TinyStoriesTransformer(
    vocab_size=vocab_size,
    d_model=256,
    n_heads=8,
    d_ff=1024,
    n_layers=4,
    max_len=MAX_LENGTH
).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parámetros entrenables totales: {total_params:,}")

# 3. Creamos un batch sintético para probar que la arquitectura compila y el forward pass funciona
# Dimensiones: [Batch_Size=128, Secuencia=64]
input_batch = torch.randint(0, vocab_size, (128, MAX_LENGTH)).to(device)
labels_batch = torch.randint(0, vocab_size, (128, MAX_LENGTH)).to(device)

logits, loss = model(input_batch, targets=labels_batch)
print(f"Forma de los logits de salida: {logits.shape}")
print(f"Pérdida inicial pre-entrenamiento: {loss.item():.4f}")