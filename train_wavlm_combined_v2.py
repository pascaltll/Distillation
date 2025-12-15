# =============================================================================
# SCRIPT 2C: FINE-TUNING WAVLM CON TRAINER + COMET ML + REGULARIZACIÓN + METRICAS AVANZADAS
# =============================================================================

# 1. IMPORTAR COMET_ML PRIMERO (Antes que torch o transformers)
import comet_ml
import os

# Configuración de Comet
os.environ["COMET_API_KEY"] = "d9I6nCchD7WvERsLUr64f7izh"
os.environ["COMET_PROJECT_NAME"] = "wavML"
os.environ["COMET_WORKSPACE"] = "pascaltll"

# Iniciar el experimento
comet_ml.init()

import torch
import torch.nn as nn
import pandas as pd
import librosa
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from transformers import (
    WavLMForSequenceClassification,
    AutoFeatureExtractor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

# ------------------------------------------------
# 1. CONFIGURACIÓN
# ------------------------------------------------
class Config:
    TRAIN_CSV = "processed_combined_teacher/combined_train.csv"
    VAL_CSV = "processed_combined_teacher/combined_val.csv"
    MODEL_NAME = "microsoft/wavlm-base-plus"
    OUTPUT_DIR = Path("wavlm_comb_data_20_eph") # Carpeta de salida actualizada
    SAMPLE_RATE = 16000
    MAX_DURATION = 12.0
    TARGET_LEN = int(SAMPLE_RATE * MAX_DURATION)
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 25     # Aumentado a 20 épocas
    BATCH_SIZE_PER_GPU = 8 

cfg = Config()
cfg.OUTPUT_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------
# 2. CARGA DE DATOS Y PESOS
# ------------------------------------------------
try:
    train_df = pd.read_csv(cfg.TRAIN_CSV)
    val_df = pd.read_csv(cfg.VAL_CSV)
except FileNotFoundError:
    print(f"❌ ERROR: No se encontraron los archivos CSV.")
    exit()

label_map = {'ang': 0, 'hap': 1, 'sad': 2, 'neu': 3}
counts = train_df['emotion'].value_counts()

# Cálculo de pesos para balancear clases
class_weights = 1.0 / counts
class_weights = class_weights / class_weights.sum() * len(label_map)

weight_tensor = torch.zeros(len(label_map), dtype=torch.float)
for emo, idx in label_map.items():
    if emo in class_weights.index:
        weight_tensor[idx] = class_weights.loc[emo]

print("⚖️  Pesos de clase calculados:", weight_tensor)

# ------------------------------------------------
# 3. CUSTOM TRAINER (Soporte para pesos de clase)
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
        
        # Carga y procesado de audio
        y, sr = librosa.load(wav_path, sr=self.config.SAMPLE_RATE)
        if len(y) > self.config.TARGET_LEN: 
            y = y[:self.config.TARGET_LEN]
        else: 
            y = np.pad(y, (0, self.config.TARGET_LEN - len(y)), mode='constant')
            
        inputs = self.processor(y, sampling_rate=self.config.SAMPLE_RATE, return_tensors="pt")
        input_values = inputs.input_values.squeeze(0).numpy()
        
        label = self.label_map.get(row['emotion'], 3)
        return {"input_values": input_values, "labels": label}

processor = AutoFeatureExtractor.from_pretrained(cfg.MODEL_NAME)
train_dataset = IEMOCAPDataset(train_df, processor, cfg)
val_dataset = IEMOCAPDataset(val_df, processor, cfg)

# ------------------------------------------------
# 5. MÉTRICAS AVANZADAS PARA COMET
# ------------------------------------------------
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    
    # Métricas básicas
    acc = accuracy_score(labels, preds)
    uar = recall_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')
    
    # Logging avanzado a Comet ML
    experiment = comet_ml.get_global_experiment()
    if experiment:
        # 1. Matriz de Confusión Interactiva
        experiment.log_confusion_matrix(
            y_true=labels,
            y_predicted=preds,
            labels=['ang', 'hap', 'sad', 'neu'],
            title="Confusion Matrix (Validation)"
        )
        
        # 2. Accuracy por clase (para ver qué emoción falla)
        cm = confusion_matrix(labels, preds)
        # Evitar división por cero
        with np.errstate(divide='ignore', invalid='ignore'):
            class_acc = cm.diagonal() / cm.sum(axis=1)
            class_acc = np.nan_to_num(class_acc) # Reemplazar NaNs con 0
            
        for name, score in zip(['ang', 'hap', 'sad', 'neu'], class_acc):
            experiment.log_metric(f"val_acc_{name}", score)
            
        # 3. Métricas adicionales
        experiment.log_metric("val_f1_macro", f1)

    return {"accuracy": acc, "uar": uar, "f1": f1}

# ------------------------------------------------
# 6. MODELO Y ENTRENAMIENTO
# ------------------------------------------------
model = WavLMForSequenceClassification.from_pretrained(cfg.MODEL_NAME, num_labels=4)
model.to(DEVICE)

# A. Congelar Encoder (Estabilidad)
model.freeze_feature_encoder()
print("❄️  Feature Encoder congelado.")

training_args = TrainingArguments(
    output_dir=str(cfg.OUTPUT_DIR),
    
    # --- Configuración para COMET ML ---
    report_to=["comet_ml"], 
    run_name="wavlm-combined-20ep-advanced",
    
    # --- Regularización y Batch ---
    per_device_train_batch_size=cfg.BATCH_SIZE_PER_GPU, # 8
    per_device_eval_batch_size=cfg.BATCH_SIZE_PER_GPU,
    gradient_accumulation_steps=4, # Simula batch efectivo de 32
    
    # --- Optimizador ---
    learning_rate=cfg.LEARNING_RATE,
    weight_decay=0.01,   # Regularización L2
    warmup_ratio=0.1,    # Calentamiento del LR
    max_grad_norm=1.0,   # Clipping de gradiente
    
    # --- Duración y Guardado ---
    num_train_epochs=cfg.NUM_EPOCHS, # 20 Épocas
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="uar",
    greater_is_better=True,
    
    logging_steps=50,
    save_total_limit=2,
)

trainer = CustomTrainer(
    class_weights=weight_tensor,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    # Paciencia de 4 épocas (un poco más permisivo ya que aumentamos a 20 épocas)
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)] 
)

print("\n🚀 INICIANDO ENTRENAMIENTO (20 ÉPOCAS + REGULARIZACIÓN + METRICAS EXTRA)")
trainer.train()

# Guardado final
trainer.save_model(cfg.OUTPUT_DIR)
processor.save_pretrained(cfg.OUTPUT_DIR)
print(f"\n✅ Modelo guardado en: {cfg.OUTPUT_DIR}")
print("🔍 Revisa Comet ML para ver la Matriz de Confusión y métricas por clase.")
