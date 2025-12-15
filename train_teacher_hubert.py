# =============================================================================
# SCRIPT: MAESTRO 2 - HuBERT BASE (LA APUESTA SEGURA)
# =============================================================================

import comet_ml
import os

# Configuración de Comet
os.environ["COMET_API_KEY"] = "d9I6nCchD7WvERsLUr64f7izh"
os.environ["COMET_PROJECT_NAME"] = "Teacher-HuBERT-Base" 
os.environ["COMET_WORKSPACE"] = "pascaltll"

comet_ml.init()

import torch
import torch.nn as nn
import pandas as pd
import librosa
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from transformers import (
    HubertForSequenceClassification,
    AutoFeatureExtractor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

# ------------------------------------------------
# 1. CONFIGURACIÓN (CAMBIO A MODELO BASE)
# ------------------------------------------------
class Config:
    TRAIN_CSV = "processed_combined_teacher/combined_train.csv"
    VAL_CSV = "processed_combined_teacher/combined_val.csv"
    
    # --- CAMBIO CLAVE: MODELO BASE ---
    # Este es el equivalente al WavLM Base que funcionó bien.
    MODEL_NAME = "facebook/hubert-base-ls960" 
    
    OUTPUT_DIR = Path("Huber-base-v1")
    
    SAMPLE_RATE = 16000
    MAX_DURATION = 10.0
    TARGET_LEN = int(SAMPLE_RATE * MAX_DURATION)
    
    # --- CONFIGURACIÓN OPTIMIZADA (Igual a tu WavLM ganador) ---
    BATCH_SIZE_PER_GPU = 8        # Cabe perfectamente en GPU
    GRADIENT_ACCUMULATION = 4     # 8 * 4 = 32 Batch efectivo
    
    LEARNING_RATE = 2e-5          # LR estándar para Base
    NUM_EPOCHS = 20               # Damos tiempo, el Early Stopping cortará si es necesario
    
cfg = Config()
cfg.OUTPUT_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------
# 2. CARGA DE DATOS
# ------------------------------------------------
try:
    train_df = pd.read_csv(cfg.TRAIN_CSV)
    val_df = pd.read_csv(cfg.VAL_CSV)
except FileNotFoundError:
    print(f"❌ ERROR: No se encontraron los CSVs.")
    exit()

label_map = {'ang': 0, 'hap': 1, 'sad': 2, 'neu': 3}
counts = train_df['emotion'].value_counts()
class_weights = 1.0 / counts
class_weights = class_weights / class_weights.sum() * len(label_map)

weight_tensor = torch.zeros(len(label_map), dtype=torch.float)
for emo, idx in label_map.items():
    if emo in class_weights.index: weight_tensor[idx] = class_weights.loc[emo]

print("⚖️ Pesos:", weight_tensor)

# ------------------------------------------------
# 3. CUSTOM TRAINER
# ------------------------------------------------
class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.model.device) if class_weights is not None else None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# ------------------------------------------------
# 4. DATASET
# ------------------------------------------------
class IEMOCAPDataset(torch.utils.data.Dataset):
    def __init__(self, df, processor, config):
        self.df = df
        self.processor = processor
        self.config = config
        self.label_map = {'ang': 0, 'hap': 1, 'sad': 2, 'neu': 3}
    
    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path = row['wav_path']
        try:
            y, sr = librosa.load(wav_path, sr=self.config.SAMPLE_RATE)
        except:
            y = np.zeros(self.config.TARGET_LEN)
            
        if len(y) > self.config.TARGET_LEN: y = y[:self.config.TARGET_LEN]
        else: y = np.pad(y, (0, self.config.TARGET_LEN - len(y)), mode='constant')
            
        inputs = self.processor(y, sampling_rate=self.config.SAMPLE_RATE, return_tensors="pt")
        input_values = inputs.input_values.squeeze(0).numpy()
        label = self.label_map.get(row['emotion'], 3)
        return {"input_values": input_values, "labels": label}

processor = AutoFeatureExtractor.from_pretrained(cfg.MODEL_NAME)
train_dataset = IEMOCAPDataset(train_df, processor, cfg)
val_dataset = IEMOCAPDataset(val_df, processor, cfg)

# ------------------------------------------------
# 5. MÉTRICAS
# ------------------------------------------------
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    uar = recall_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')
    
    experiment = comet_ml.get_global_experiment()
    if experiment:
        experiment.log_confusion_matrix(y_true=labels, y_predicted=preds, labels=['ang', 'hap', 'sad', 'neu'])
        cm = confusion_matrix(labels, preds)
        with np.errstate(divide='ignore', invalid='ignore'):
            class_acc = np.nan_to_num(cm.diagonal() / cm.sum(axis=1))
        for name, score in zip(['ang', 'hap', 'sad', 'neu'], class_acc):
            experiment.log_metric(f"val_acc_{name}", score)
            
    return {"accuracy": acc, "uar": uar, "f1": f1}

# ------------------------------------------------
# 6. MODELO & ESTRATEGIA (LA FORMULA GANADORA)
# ------------------------------------------------
print(f"🔄 Descargando {cfg.MODEL_NAME}...")
model = HubertForSequenceClassification.from_pretrained(cfg.MODEL_NAME, num_labels=4)
model.to(DEVICE)

print("\n🔓 APLICANDO ESTRATEGIA GANADORA (Partial Unfreeze - Base)...")

# 1. Congelar Feature Encoder (CNNs)
model.freeze_feature_encoder()

# 2. Congelar Transformer Base
for param in model.hubert.parameters():
    param.requires_grad = False

# 3. DESCONGELAR ÚLTIMAS 2 CAPAS (Estrategia V4 de WavLM)
# HuBERT Base tiene 12 capas. Descongelamos la 10 y 11.
print("   -> Descongelando últimas 2 capas del Encoder...")
for param in model.hubert.encoder.layers[-2:].parameters():
    param.requires_grad = True

# 4. Clasificador activo
for param in model.classifier.parameters():
    param.requires_grad = True
if hasattr(model, 'projector'):
    for param in model.projector.parameters():
        param.requires_grad = True

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ HuBERT Base Listo. Parámetros entrenables: {trainable_params:,}")

# ------------------------------------------------
# 7. ENTRENAMIENTO
# ------------------------------------------------
training_args = TrainingArguments(
    output_dir=str(cfg.OUTPUT_DIR),
    report_to=["comet_ml"], 
    run_name="hubert-base-teacher", # Nombre del experimento
    
    per_device_train_batch_size=cfg.BATCH_SIZE_PER_GPU, 
    per_device_eval_batch_size=cfg.BATCH_SIZE_PER_GPU,
    gradient_accumulation_steps=cfg.GRADIENT_ACCUMULATION,
    
    learning_rate=cfg.LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.1,
    max_grad_norm=1.0, 
    
    num_train_epochs=cfg.NUM_EPOCHS,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="uar",
    greater_is_better=True,
    logging_steps=50,
    save_total_limit=1,
)

trainer = CustomTrainer(
    class_weights=weight_tensor,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
)

print("\n🚀 INICIANDO ENTRENAMIENTO (HuBERT BASE)")
trainer.train()

trainer.save_model(cfg.OUTPUT_DIR)
processor.save_pretrained(cfg.OUTPUT_DIR)
print(f"\n✅ Maestro HuBERT Base guardado en: {cfg.OUTPUT_DIR}")
