"""
grokking_optimizer_ablation.py

Executes an optimizer ablation study comparing AdamW, standard Adam, and SGD.
Evaluates the training and spectral dynamics to empirically demonstrate that 
the geometric pressure from decoupled weight decay is a strictly necessary 
condition to traverse the topological barrier and induce grokking.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import itertools

# ==========================================
# 1. GENERACIÓN DEL DATASET ALGORÍTMICO
# ==========================================
class ModularAdditionDataset(Dataset):
    def __init__(self, p=97, split='train', train_ratio=0.5, seed=42):
        """
        Dataset de aritmética modular a + b = c (mod p).
        Vocabulario (0 a p-1) para enteros, p para '+', p+1 para '='.
        """
        self.p = p
        torch.manual_seed(seed) # Mantenemos la semilla estática para control
        
        # Generar universo de ecuaciones (p^2)
        all_pairs = torch.cartesian_prod(torch.arange(p), torch.arange(p))
        indices = torch.randperm(len(all_pairs))
        all_pairs = all_pairs[indices]
        
        # Split 50/50
        split_idx = int(len(all_pairs) * train_ratio)
        self.data = all_pairs[:split_idx] if split == 'train' else all_pairs[split_idx:]
        
        self.plus_token = p
        self.eq_token = p + 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        a, b = self.data[idx]
        c = (a + b) % self.p
        # Tensor X: [a, +, b, =]
        x = torch.tensor([a, self.plus_token, b, self.eq_token], dtype=torch.long)
        # Tensor Y (Shifted target): [+, b, =, c]
        y = torch.tensor([self.plus_token, b, self.eq_token, c], dtype=torch.long)
        return x, y

# ==========================================
# 2. ARQUITECTURA TRANSFORMER CAUSAL
# ==========================================
class SmallCausalTransformer(nn.Module):
    def __init__(self, vocab_size=100, d_model=128, n_heads=4, n_layers=2, d_ff=512):
        """
        Transformer configurado con N ~ 422,000 parámetros y Pre-LN.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 4, d_model)) # seq_len = L = 4
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_ff,
            norm_first=True, # Pre-LN configuración
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # Inicialización Xavier Estricta
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding
        # Máscara causal triangular superior
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
        out = self.transformer(x, mask=mask, is_causal=True)
        return self.lm_head(out)

# ==========================================
# 3. EXTRACCIÓN ESPECTRAL (HVP + POWER ITERATION)
# ==========================================
def compute_lambda_max(model, loss_fn, x, y, num_iterations=20):
    """
    Aproximación iterativa de Lambda Max usando Productos Hessiano-Vector.
    """
    # Desactivar explícitamente kernels de atención fusionada (FlashAttention)
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        model.zero_grad()
        outputs = model(x)
        
        # Evaluado exclusivamente en el último token de la secuencia
        loss = loss_fn(outputs[:, -1, :], y[:, -1]) 
        
        params = [p for p in model.parameters() if p.requires_grad]
        
        # Primera derivada (Gradiente)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        
        # Inicializar vector aleatorio v0
        v = [torch.randn_like(p) for p in params]
        v_norm = torch.sqrt(sum((x**2).sum() for x in v))
        v = [x / v_norm for x in v]
        
        # Método de la potencia
        for _ in range(num_iterations):
            grad_v = sum((g * v_i).sum() for g, v_i in zip(grads, v))
            hvp = torch.autograd.grad(grad_v, params, retain_graph=True) #
            
            hvp_norm = torch.sqrt(sum((x**2).sum() for x in hvp))
            v = [x / (hvp_norm + 1e-8) for x in hvp] # Prevenir overflow
            
        # Cociente de Rayleigh
        grad_v = sum((g * v_i).sum() for g, v_i in zip(grads, v))
        final_hvp = torch.autograd.grad(grad_v, params)
        lambda_max = sum((v_i * h_i).sum() for v_i, h_i in zip(v, final_hvp))
        
        return lambda_max.item()

# ==========================================
# 4. BUCLE DE ENTRENAMIENTO PRINCIPAL
# ==========================================
def train_model(optimizer_name, device):
    print(f"\n--- Iniciando Experimento: {optimizer_name} ---")
    
    # Preparación de datos
    train_dataset = ModularAdditionDataset(split='train')
    val_dataset = ModularAdditionDataset(split='val')
    # Batch size estático de B=256
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    # Reiniciar semilla para asegurar inicialización idéntica de pesos
    torch.manual_seed(42)
    model = SmallCausalTransformer().to(device)
    loss_fn = nn.CrossEntropyLoss()
    
    # Selección del Optimizador
    lr = 1e-3
    weight_decay = 1.0 # Catalizador del grokking
    
    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98)) #
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

    steps = 25000 # Horizonte asintótico de entrenamiento
    eval_interval = 500
    
    history = {'step': [], 'lambda_max': [], 'val_acc': []}
    step_iterator = iter(train_loader)
    
    for step in range(1, steps + 1):
        try:
            x, y = next(step_iterator)
        except StopIteration:
            step_iterator = iter(train_loader)
            x, y = next(step_iterator)
            
        x, y = x.to(device), y.to(device)
        
        model.train()
        optimizer.zero_grad()
        outputs = model(x)
        loss = loss_fn(outputs[:, -1, :], y[:, -1])
        loss.backward()
        optimizer.step()
        
        # Evaluación periódica (Monitoreo Activo)
        if step % eval_interval == 0 or step == 1:
            # 1. Calcular Accuracy
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    v_out = model(vx)
                    preds = torch.argmax(v_out[:, -1, :], dim=-1)
                    correct += (preds == vy[:, -1]).sum().item()
                    total += vy.size(0)
            val_acc = correct / total
            
            # 2. Calcular Lambda Max
            model.train() # Requiere requires_grad=True
            l_max = compute_lambda_max(model, loss_fn, x, y)
            
            history['step'].append(step)
            history['val_acc'].append(val_acc)
            history['lambda_max'].append(l_max)
            print(f"Paso {step:05d} | Val Acc: {val_acc:.4f} | Lambda_Max: {l_max:.2f}")
            
    return history

# ==========================================
# 5. EJECUCIÓN Y VISUALIZACIÓN DE RESULTADOS
# ==========================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Ejecutando en hardware: {device}")
    
    optimizers = ["AdamW", "Adam", "SGD"]
    results = {}
    
    for opt in optimizers:
        results[opt] = train_model(opt, device)
        
    # Graficar y guardar figura para el Apéndice A
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Precisión de Validación
    plt.subplot(2, 1, 1)
    for opt in optimizers:
        plt.plot(results[opt]['step'], results[opt]['val_acc'], label=f'{opt} (Val Acc)')
    plt.title('Ablation Study: Algorithmic Generalization (Accuracy)')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Evolución Espectral (Hessian Lambda Max)
    plt.subplot(2, 1, 2)
    for opt in optimizers:
        plt.plot(results[opt]['step'], results[opt]['lambda_max'], label=f'{opt} (\u03bb_max)', alpha=0.8)
    plt.title('Spectral Dynamics: Impact of Coupled vs. Decoupled Weight Decay')
    plt.xlabel('Optimization Steps')
    plt.ylabel('Dominant Eigenvalue (\u03bb_max)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimizer_ablation_results.png', dpi=300)
    print("\nExperimento finalizado. Gráfica guardada como 'optimizer_ablation_results.png'.")