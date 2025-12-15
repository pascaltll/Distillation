# =============================================================================
# SCRIPT: FINE-TUNING UNISPEECH-SAT LARGE (ESTRATEGIA WAVLM V2)
# =============================================================================

import comet_ml
import os
import torch
import torch.nn as nn
import pandas as pd
import librosa
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from transformers import (
    UniSpeechSatForSequenceClassification,  # 🛑 CAMBIO CRÍTICO: Modelo SAT
    AutoFeatureExtractor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

# Configuración de Comet (Asegúrate de cambiar el nombre del proyecto)
os.environ["COMET_API_KEY"] = "d9I6nCchD7WvERsLUr64f7izh"
os.environ["COMET_PROJECT_NAME"] = "UniSpeech-SAT-Large" # 🛑 NUEVO PROYECTO
os.environ["COMET_WORKSPACE"] = "pascaltll"

comet_ml.init()

# ------------------------------------------------
# 1. CONFIGURACIÓN (ADAPTADA A UNISPEECH-SAT)
# ------------------------------------------------
class Config:
    TRAIN_CSV = "processed_combined_teacher/combined_train.csv"
    VAL_CSV = "processed_combined_teacher/combined_val.csv"

    MODEL_NAME = "microsoft/unispeech-sat-large" # 🛑 CAMBIO CRÍTICO: Modelo SAT Large
    OUTPUT_DIR = Path("unispeech_sat_large_finetuned")

    SAMPLE_RATE = 16000
    MAX_DURATION = 8.0
    TARGET_LEN = int(SAMPLE_RATE * MAX_DURATION)

    # Conservamos tu configuración optimizada para VRAM
    BATCH_SIZE_PER_GPU = 16
    GRADIENT_ACCUMULATION = 2
    
    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 15

cfg = Config()
cfg.OUTPUT_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------
# 2. CARGA DE DATOS Y PESOS (Idéntico a tu script)
# ------------------------------------------------
try:
    train_df = pd.read_csv(cfg.TRAIN_CSV)
    val_df = pd.read_csv(cfg.VAL_CSV)
except FileNotFoundError:
    print(f"❌ ERROR: No se encontraron los archivos CSV.")
    exit()

label_map = {'ang': 0, 'hap': 1, 'sad': 2, 'neu': 3}
counts = train_df['emotion'].value_counts()
class_weights = 1.0 / counts
class_weights = class_weights / class_weights.sum() * len(label_map)

weight_tensor = torch.zeros(len(label_map), dtype=torch.float)
for emo, idx in label_map.items():
    if emo in class_weights.index:
        weight_tensor[idx] = class_weights.loc[emo]

print("⚖️ Pesos de clase:", weight_tensor)

# ------------------------------------------------
# 3. CUSTOM TRAINER (Idéntico a tu script)
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
# 4. DATASET (Idéntico a tu script)
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
        except Exception as e:
            print(f"Error cargando {wav_path}: {e}")
            y = np.zeros(self.config.TARGET_LEN)

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
# 5. MÉTRICAS AVANZADAS (COMET) (Idéntico a tu script)
# ------------------------------------------------
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids

    acc = accuracy_score(labels, preds)
    uar = recall_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')

    experiment = comet_ml.get_global_experiment()
    if experiment:
        experiment.log_confusion_matrix(
            y_true=labels,
            y_predicted=preds,
            labels=['ang', 'hap', 'sad', 'neu'],
            title="Confusion Matrix (Validation)"
        )

        cm = confusion_matrix(labels, preds)
        with np.errstate(divide='ignore', invalid='ignore'):
            class_acc = cm.diagonal() / cm.sum(axis=1)
            class_acc = np.nan_to_num(class_acc)

        for name, score in zip(['ang', 'hap', 'sad', 'neu'], class_acc):
            experiment.log_metric(f"val_acc_{name}", score)

    return {"accuracy": acc, "uar": uar, "f1": f1}

# ------------------------------------------------
# 6. MODELO SAT LARGE & ESTRATEGIA DE DESCONGELADO PARCIAL
# ------------------------------------------------
print(f"🔄 Descargando {cfg.MODEL_NAME}...")
# 🛑 CAMBIO CRÍTICO: Cargar la clase de UniSpeech-SAT
model = UniSpeechSatForSequenceClassification.from_pretrained(cfg.MODEL_NAME, num_labels=4)
model.to(DEVICE)

print("\n🔓 APLICANDO ESTRATEGIA: PARTIAL UNFREEZE (6 Capas)...")

# La estructura de capas es la misma que WavLM/Wav2Vec 2.0
# 1. Congelar Feature Encoder (CNNs)
model.freeze_feature_encoder()

# 2. Congelar TODO el Transformer primero
for param in model.unispeech_sat.parameters(): # 🛑 CAMBIO: Usar 'unispeech_sat'
    param.requires_grad = False

# 3. DESCONGELAR las últimas 6 capas (Layers 18 a 23)
print("   -> Descongelando capas 18, 19, 20, 21, 22, 23 del Encoder...")
for param in model.unispeech_sat.encoder.layers[-6:].parameters(): # 🛑 CAMBIO: Usar 'unispeech_sat'
    param.requires_grad = True

# 4. Asegurar que el clasificador se entrena
for param in model.classifier.parameters():
    param.requires_grad = True
if hasattr(model, 'projector'):
    for param in model.projector.parameters():
        param.requires_grad = True

# Verificar parámetros
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"✅ Configuración de UniSpeech-SAT Large lista.")
print(f"   Parámetros totales: {total_params:,}")
print(f"   Parámetros entrenables: {trainable_params:,} ({(trainable_params/total_params)*100:.1f}%)")

# ------------------------------------------------
# 7. ENTRENAMIENTO
# ------------------------------------------------
training_args = TrainingArguments(
    output_dir=str(cfg.OUTPUT_DIR),
    report_to=["comet_ml"],
    run_name="unispeech-sat-LARGE-unfrozen-6layers",

    # Batch & Steps (Batch Efectivo de 32)
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

    logging_steps=20,
    save_total_limit=1,
    fp16=True, # Obligatorio para ahorrar memoria con el modelo Large
    dataloader_num_workers=0 # Estabilidad
)

trainer = CustomTrainer(
    class_weights=weight_tensor,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

print("\n🚀 INICIANDO ENTRENAMIENTO (UNISPEECH-SAT LARGE)")
trainer.train()

trainer.save_model(cfg.OUTPUT_DIR)
processor.save_pretrained(cfg.OUTPUT_DIR)
print(f"\n✅ Modelo UniSpeech-SAT Large guardado en: {cfg.OUTPUT_DIR}")
