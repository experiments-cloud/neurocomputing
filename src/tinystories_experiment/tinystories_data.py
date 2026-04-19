"""
tinystories_data.py

Downloads and tokenizes the TinyStories natural language corpus.
Utilizes the GPT-2 tokenizer and enforces a strict sequence length 
truncation (MAX_LENGTH = 64) to maintain memory stability during 
the computationally intensive Hessian-Vector Product calculations.
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# ==========================================
# Configuración de Hiperparámetros del Dataset
# ==========================================
TRAIN_SAMPLES = 100000  # Subconjunto de entrenamiento
VAL_SAMPLES = 5000      # Subconjunto de validación
MAX_LENGTH = 64         # Longitud de contexto estricta para controlar RAM en HVP
BATCH_SIZE = 128        # Ajustable según tu GPU

print("1. Descargando el dataset TinyStories...")
# Descargamos el dataset completo desde Hugging Face
dataset = load_dataset("roneneldan/TinyStories")

# Seleccionamos los subconjuntos para no entrenar durante días
train_subset = dataset["train"].select(range(TRAIN_SAMPLES))
val_subset = dataset["validation"].select(range(VAL_SAMPLES))

print(f"Dataset cargado: {len(train_subset)} train, {len(val_subset)} val.")

print("2. Configurando el Tokenizador...")
# Usamos el tokenizador de GPT-2 (BPE) por su vocabulario eficiente (~50k tokens)
# Es un estándar en la literatura y evita tener que entrenar uno desde cero.
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token # GPT-2 no tiene pad_token por defecto

# ==========================================
# Función de Tokenización
# ==========================================
def tokenize_function(examples):
    # Truncamos y rellenamos a MAX_LENGTH para tener tensores uniformes
    # Esto es crucial para estabilizar el cálculo del eigenvalor dominante
    tokens = tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    
    # En modelado autorregresivo (Causal LM), los labels son los mismos input_ids
    # El modelo se encargará de hacer el shift (desplazamiento) internamente o 
    # en la función de pérdida.
    tokens["labels"] = tokens["input_ids"].clone()
    return tokens

print("3. Aplicando tokenización (esto puede tomar un momento)...")
# Mapeamos la función sobre los datasets
tokenized_train = train_subset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_val = val_subset.map(tokenize_function, batched=True, remove_columns=["text"])

# Convertimos a formato PyTorch
tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

print("4. Creando PyTorch DataLoaders...")
# Dataloaders listos para el ciclo de entrenamiento
train_dataloader = DataLoader(tokenized_train, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(tokenized_val, batch_size=BATCH_SIZE, shuffle=False)

print("¡Paso 1 Completado! Datos listos para el Transformer.")

# Comprobación rápida de las dimensiones del tensor
sample_batch = next(iter(train_dataloader))
print(f"Forma de input_ids: {sample_batch['input_ids'].shape}") # Debería ser [128, 64]