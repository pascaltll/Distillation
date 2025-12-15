# =============================================================================
# SCRIPT 2C: FINE-TUNING WAVLM CON TRAINER + LOSS PONDERADA (BALANCEO AUTOMÁTICO)
# =============================================================================

import os
import torch
import torch.nn as nn
import pandas as pd
import librosa
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, recall_score
from transformers import (
    WavLMForSequenceClassification,
    AutoFeatureExtractor,
    TrainingArguments,
    Trainer
)

# ------------------------------------------------
# 1. CONFIGURACIÓN
# ------------------------------------------------
class Config:
    TRAIN_CSV = "processed_iemocap/iemocap_train.csv"
    VAL_CSV = "processed_iemocap/iemocap_val.csv"

    MODEL_NAME = "microsoft/wavlm-base-plus"
    OUTPUT_DIR = Path("wavlm_finetuned_balanced")

    SAMPLE_RATE = 16000
    MAX_DURATION = 12.0
    TARGET_LEN = int(SAMPLE_RATE * MAX_DURATION)

    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 15
    BATCH_SIZE_PER_GPU = 8

cfg = Config()
cfg.OUTPUT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------
# 2. CÁLCULO DE PESOS DE CLASE
# ------------------------------------------------
try:
    train_df = pd.read_csv(cfg.TRAIN_CSV)
    val_df = pd.read_csv(cfg.VAL_CSV)
except FileNotFoundError:
    print("\n❌ ERROR: ¡Asegúrate de ejecutar el script de preprocesamiento primero!")
    exit()

label_map = {'ang': 0, 'hap': 1, 'sad': 2, 'neu': 3}
counts = train_df['emotion'].value_counts()
print("\nDistribución de clases en Train:")
print(counts)

class_weights = 1.0 / counts
class_weights = class_weights / class_weights.sum() * len(label_map)

weight_tensor = torch.zeros(len(label_map), dtype=torch.float)
for emo, idx in label_map.items():
    if emo in class_weights.index:
        weight_tensor[idx] = class_weights.loc[emo]

print("\n⚙️ Pesos de Clase Calculados (en CPU):")
print(weight_tensor)
print("-" * 50)

# ------------------------------------------------
# 3. CUSTOM TRAINER
# ------------------------------------------------
class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = class_weights.to(self.model.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

# ------------------------------------------------
# 4. DATASET y MÉTRICAS
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

        y, sr = librosa.load(wav_path, sr=self.config.SAMPLE_RATE)
        if len(y) > self.config.TARGET_LEN: y = y[:self.config.TARGET_LEN]
        else: y = np.pad(y, (0, self.config.TARGET_LEN - len(y)), mode='constant')

        inputs = self.processor(y, sampling_rate=self.config.SAMPLE_RATE, return_tensors="pt")
        input_values = inputs.input_values.squeeze(0).numpy()

        label = self.label_map.get(row['emotion'], 3)

        return {"input_values": input_values, "labels": label}

train_dataset = IEMOCAPDataset(train_df, AutoFeatureExtractor.from_pretrained(cfg.MODEL_NAME), cfg)
val_dataset = IEMOCAPDataset(val_df, AutoFeatureExtractor.from_pretrained(cfg.MODEL_NAME), cfg)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    wa = accuracy_score(labels, preds)
    uar = recall_score(labels, preds, average='macro')
    return {"accuracy": wa, "uar": uar}

# ------------------------------------------------
# 5. ENTRENAMIENTO
# ------------------------------------------------
training_args = TrainingArguments(
    output_dir=str(cfg.OUTPUT_DIR),
    per_device_train_batch_size=cfg.BATCH_SIZE_PER_GPU,
    per_device_eval_batch_size=cfg.BATCH_SIZE_PER_GPU,
    num_train_epochs=cfg.NUM_EPOCHS,
    learning_rate=cfg.LEARNING_RATE,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="uar",
    greater_is_better=True,
    logging_steps=50,
    save_total_limit=1,
    report_to="none",
    # Opcional: para eliminar el warning de find_unused_parameters
    # ddp_find_unused_parameters=False,
)

model = WavLMForSequenceClassification.from_pretrained(cfg.MODEL_NAME, num_labels=4)

trainer = CustomTrainer(
    class_weights=weight_tensor,   # ← IMPORTANTE
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("\n🔥 INICIANDO FINE-TUNING CON PÉRDIDA PONDERADA Y MULTI-GPU")
print("==================================================")

trainer.train()

trainer.save_model(cfg.OUTPUT_DIR)
AutoFeatureExtractor.from_pretrained(cfg.MODEL_NAME).save_pretrained(cfg.OUTPUT_DIR)
