"""
tinystories_data_ablated.py

Semantic Collapse Ablation Study.
Downloads the TinyStories corpus but intercepts the text stream before tokenization 
to deterministically collapse synonyms into single root tokens. This forces strict 
syntactic rigidity to test the causal effect of semantic flexibility on the 
Hessian's dominant eigenvalue.
"""

import torch
import re
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# ==========================================
# Hyperparameter Configuration
# ==========================================
TRAIN_SAMPLES = 100000  
VAL_SAMPLES = 5000      
MAX_LENGTH = 64         
BATCH_SIZE = 128        

print("1. Downloading the TinyStories dataset...")
dataset = load_dataset("roneneldan/TinyStories")
train_subset = dataset["train"].select(range(TRAIN_SAMPLES))
val_subset = dataset["validation"].select(range(VAL_SAMPLES))

print("2. Configuring the Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token 

# ==========================================
# SEMANTIC COLLAPSE DICTIONARY (Ablation)
# ==========================================
# Forcing synonym groups into a single root token (Rigidity)
SYNONYM_MAP = {
    r'\b(huge|giant|massive|enormous|large)\b': 'big',
    r'\b(tiny|little|miniature)\b': 'small',
    r'\b(glad|joyful|cheerful|delighted)\b': 'happy',
    r'\b(mad|furious|upset)\b': 'angry',
    r'\b(quick|rapid|swift)\b': 'fast',
    r'\b(speak|shout|whisper|yell|exclaim)\b': 'say',
    r'\b(leap|bound|hop)\b': 'jump',
    r'\b(beautiful|pretty|gorgeous|lovely)\b': 'nice',
    r'\b(scared|terrified|frightened)\b': 'afraid',
    r'\b(kids|toddlers|youth)\b': 'children',
    r'\b(mom|mommy|mother)\b': 'mama',
    r'\b(dad|daddy|father)\b': 'papa'
}

def semantic_collapse_function(examples):
    collapsed_texts = []
    
    for text in examples["text"]:
        # Normalize to lowercase to ensure regex catches all instances
        text = text.lower()
        
        # Apply semantic collapse iteratively
        for pattern, root_word in SYNONYM_MAP.items():
            text = re.sub(pattern, root_word, text)
            
        collapsed_texts.append(text)
        
    # Tokenize the already collapsed texts
    tokens = tokenizer(
        collapsed_texts, 
        padding="max_length", 
        truncation=True, 
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    
    # In autoregressive modeling, labels are the input_ids
    tokens["labels"] = tokens["input_ids"].clone()
    return tokens

print("3. Applying Semantic Collapse and Tokenization...")
tokenized_train = train_subset.map(semantic_collapse_function, batched=True, remove_columns=["text"])
tokenized_val = val_subset.map(semantic_collapse_function, batched=True, remove_columns=["text"])

tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

print("4. Creating PyTorch DataLoaders (Ablated)...")
train_dataloader = DataLoader(tokenized_train, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(tokenized_val, batch_size=BATCH_SIZE, shuffle=False)

print("Rigid dataset ready for extreme topology extraction!")