import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchaudio.transforms as T
import librosa
import pandas as pd
import numpy as np
import timm
from tqdm.auto import tqdm
from accelerate import Accelerator
from transformers import AutoModelForAudioClassification
import csv
import time

# =============================================================================
# ⚙️ CONFIGURACIÓN GLOBAL
# =============================================================================
CONFIG = {
    # Rutas de datos
    'train_csv': 'processed_combined_teacher/combined_train.csv',
    'val_csv':   'processed_combined_teacher/combined_val.csv',
    'results_dir': 'results',
    
    # RUTA DEL MAESTRO (Tu modelo WavLM Large finetuneado)
    'teacher_path': 'wavlm_large_finetuned_v2', 
    
    # ALUMNOS A ENTRENAR (Eficiencia SOTA)
    'students': {
        'RepVGG': 'repvgg_a0',           # Veloz en GPU
        'MobileOne': 'mobileone_s0',     # Veloz en Móvil
        'GhostNetV2': 'ghostnetv2_100'   # Mínimos FLOPs
    },
    
    # Audio Params
    'sample_rate': 16000,
    'duration': 3.0,     # 3 segundos
    'n_mels': 64,        # Altura del espectrograma
    'num_classes': 4,
    
    # Hyperparámetros de Entrenamiento
    'batch_size': 32,    # 32 por GPU * 2 GPUs = 64 Batch Efectivo
    'epochs': 20,
    'lr': 1e-4,
    
    # Hiperparámetros de Destilación
    'temp': 4.0,         # Temperatura alta para suavizar logits
    'alpha': 0.5         # 50% Maestro, 50% Ground Truth
}

# Crear directorio de resultados si no existe
os.makedirs(CONFIG['results_dir'], exist_ok=True)

# =============================================================================
# 💿 DATASET (Optimizado para Carga Rápida)
# =============================================================================
class DualInputDataset(Dataset):
    def __init__(self, csv_path, config):
        self.df = pd.read_csv(csv_path)
        self.config = config
        self.target_len = int(config['sample_rate'] * config['duration'])
        
        # Transformaciones (Spectrogram para Student)
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config['sample_rate'], n_fft=1024, hop_length=512, n_mels=config['n_mels']
        )
        self.db_transform = T.AmplitudeToDB()
        self.label_map = {'ang': 0, 'hap': 1, 'sad': 2, 'neu': 3}

    def __len__(self): return len(self.df)

    def _load_audio(self, path):
        try:
            # Librosa es más robusto para cargar diferentes formatos
            y, _ = librosa.load(path, sr=self.config['sample_rate'])
        except Exception as e:
            # Fallback silencioso (audio vacío)
            y = np.zeros(self.target_len)
            
        # Pad o Recorte
        if len(y) > self.target_len: 
            y = y[:self.target_len]
        else: 
            y = np.pad(y, (0, self.target_len - len(y)), mode='constant')
            
        return torch.tensor(y).float()

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Audio Crudo (Para el Teacher)
        raw_audio = self._load_audio(row['wav_path'])
        
        # 2. Espectrograma (Para el Student)
        mel_spec = self.mel_transform(raw_audio)
        mel_spec = self.db_transform(mel_spec)
        # Normalización simple por instancia
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)
        mel_spec = mel_spec.unsqueeze(0) # [1, 64, Time]
        
        label = self.label_map.get(row['emotion'], 3) # Default Neutro
        return raw_audio, mel_spec, torch.tensor(label, dtype=torch.long)

# =============================================================================
# 🧠 MODELO ALUMNO (TIMM Wrapper)
# =============================================================================
class StudentModel(nn.Module):
    def __init__(self, model_name, num_classes=4):
        super().__init__()
        # 'in_chans=1' adapta la primera capa para leer espectrogramas B/N
        try:
            self.backbone = timm.create_model(model_name, pretrained=True, num_classes=num_classes, in_chans=1)
        except:
            # Fallback genérico
            self.backbone = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
            
    def forward(self, x):
        return self.backbone(x)

# =============================================================================
# 📝 LOGGING (CSV)
# =============================================================================
def log_to_csv(filename, epoch, train_loss, val_acc, val_uar, mode, student_name):
    filepath = os.path.join(CONFIG['results_dir'], filename)
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Student', 'Mode', 'Epoch', 'Train_Loss', 'Val_Acc', 'Val_UAR'])
        writer.writerow([student_name, mode, epoch, f"{train_loss:.4f}", f"{val_acc:.4f}", f"{val_uar:.4f}"])

# =============================================================================
# 🔥 BUCLE DE ÉPOCA (Unificado)
# =============================================================================
def run_epoch(accelerator, student, teacher, loader, optimizer, is_train, mode="Baseline"):
    if is_train:
        student.train()
    else:
        student.eval()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    # Barra de progreso solo en el proceso principal (Rank 0)
    pbar = tqdm(loader, disable=not accelerator.is_local_main_process, leave=False, desc=f"{mode} {'Train' if is_train else 'Val'}")
    
    for raw_audio, mel_spec, labels in pbar:
        # Nota: Accelerate mueve los tensores al device automáticamente, no hace falta .to(device)
        
        with torch.set_grad_enabled(is_train):
            # Forward Student
            student_logits = student(mel_spec)
            
            # Calculo de Loss
            if mode == "Baseline":
                # Solo Cross Entropy con etiquetas reales
                loss = nn.CrossEntropyLoss()(student_logits, labels)
                
            elif mode == "Distillation":
                # Forward Teacher (Congelado)
                with torch.no_grad():
                    # WavLM necesita input raw
                    teacher_logits = teacher(raw_audio).logits
                
                # Soft Loss (Imitación)
                T = CONFIG['temp']
                soft_loss = nn.KLDivLoss(reduction="batchmean")(
                    F.log_softmax(student_logits / T, dim=1),
                    F.softmax(teacher_logits / T, dim=1)
                ) * (T**2)
                
                # Hard Loss (Realidad)
                hard_loss = nn.CrossEntropyLoss()(student_logits, labels)
                
                # Loss Total
                loss = (CONFIG['alpha'] * soft_loss) + ((1 - CONFIG['alpha']) * hard_loss)
            
            # Backward & Step
            if is_train:
                optimizer.zero_grad()
                accelerator.backward(loss) # Magia de Accelerate
                optimizer.step()
                
            total_loss += loss.item()
            
            # Recolectar métricas (Gather de todas las GPUs)
            preds = torch.argmax(student_logits, dim=1)
            all_preds.extend(accelerator.gather(preds).cpu().numpy())
            all_labels.extend(accelerator.gather(labels).cpu().numpy())
            
    # Calcular Métricas Globales
    from sklearn.metrics import recall_score, accuracy_score
    
    # Ajuste de seguridad por si el último batch tiene tamaños distintos
    min_len = min(len(all_labels), len(all_preds))
    acc = accuracy_score(all_labels[:min_len], all_preds[:min_len])
    uar = recall_score(all_labels[:min_len], all_preds[:min_len], average='macro')
    
    avg_loss = total_loss / len(loader)
    return avg_loss, acc, uar

# =============================================================================
# 🚀 FUNCIÓN PRINCIPAL
# =============================================================================
def main():
    # Inicializar Accelerate
    accelerator = Accelerator()
    
    if accelerator.is_main_process:
        print(f"\n🚀 Iniciando Experimento Multi-GPU ({accelerator.num_processes} dispositivos)")
        print(f"📂 Configuración: Batch={CONFIG['batch_size']} | Temp={CONFIG['temp']} | Alpha={CONFIG['alpha']}")

    # 1. Preparar Datos
    train_ds = DualInputDataset(CONFIG['train_csv'], CONFIG)
    val_ds = DualInputDataset(CONFIG['val_csv'], CONFIG)
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    # 2. Cargar Maestro (Si aplica)
    # Se carga fuera del bucle para no recargarlo mil veces
    if accelerator.is_main_process: print(f"🎓 Cargando Teacher desde: {CONFIG['teacher_path']}")
    
    try:
        teacher = AutoModelForAudioClassification.from_pretrained(CONFIG['teacher_path'], num_labels=CONFIG['num_classes'])
    except Exception as e:
        if accelerator.is_main_process: print(f"⚠️ Error cargando teacher local ({e}). Usando versión base de HF.")
        teacher = AutoModelForAudioClassification.from_pretrained("microsoft/wavlm-base-plus", num_labels=CONFIG['num_classes'])
    
    teacher.eval()
    for param in teacher.parameters(): param.requires_grad = False

    # 3. Bucle de Experimentación (Por cada Alumno)
    for student_name, student_arch in CONFIG['students'].items():
        if accelerator.is_main_process: 
            print(f"\n{'='*50}\n🤖 ENTRENANDO ALUMNO: {student_name} ({student_arch})\n{'='*50}")

        # --- FASE A: BASELINE (Sin ayuda) ---
        if accelerator.is_main_process: print(f"📉 Fase 1: Baseline (Sin Maestro)...")
        
        student = StudentModel(student_arch, num_classes=CONFIG['num_classes'])
        optimizer = optim.AdamW(student.parameters(), lr=CONFIG['lr'])

        # Preparar con Accelerate
        student, optimizer, train_dl, val_dl, teacher_prepped = accelerator.prepare(
            student, optimizer, train_loader, val_loader, teacher
        )
        
        best_uar = 0.0
        for epoch in range(CONFIG['epochs']):
            tl, ta, tu = run_epoch(accelerator, student, None, train_dl, optimizer, True, "Baseline")
            vl, va, vu = run_epoch(accelerator, student, None, val_dl, optimizer, False, "Baseline")
            
            if accelerator.is_main_process:
                print(f"[BL] Ep {epoch+1:02d} | Loss: {tl:.4f} | Val Acc: {va:.2%} | Val UAR: {vu:.2%}")
                log_to_csv('metrics_log.csv', epoch, tl, va, vu, 'Baseline', student_name)
                
                if vu > best_uar:
                    best_uar = vu
                    torch.save(accelerator.unwrap_model(student).state_dict(), 
                               f"{CONFIG['results_dir']}/best_baseline_{student_name}.pth")

        # Limpieza de Memoria entre fases
        del student, optimizer
        accelerator.free_memory()
        torch.cuda.empty_cache()

        # --- FASE B: DESTILACIÓN (Con Maestro) ---
        if accelerator.is_main_process: print(f"\n🧪 Fase 2: Destilación (Teacher: WavLM)...")
        
        student_dist = StudentModel(student_arch, num_classes=CONFIG['num_classes'])
        optimizer_dist = optim.AdamW(student_dist.parameters(), lr=CONFIG['lr'])
        
        student_dist, optimizer_dist, train_dl, val_dl = accelerator.prepare(
            student_dist, optimizer_dist, train_loader, val_loader
        )

        best_uar_dist = 0.0
        for epoch in range(CONFIG['epochs']):
            tl, ta, tu = run_epoch(accelerator, student_dist, teacher_prepped, train_dl, optimizer_dist, True, "Distillation")
            vl, va, vu = run_epoch(accelerator, student_dist, teacher_prepped, val_dl, optimizer_dist, False, "Distillation")
            
            if accelerator.is_main_process:
                print(f"[KD] Ep {epoch+1:02d} | Loss: {tl:.4f} | Val Acc: {va:.2%} | Val UAR: {vu:.2%}")
                log_to_csv('metrics_log.csv', epoch, tl, va, vu, 'Distillation', student_name)
                
                if vu > best_uar_dist:
                    best_uar_dist = vu
                    torch.save(accelerator.unwrap_model(student_dist).state_dict(), 
                               f"{CONFIG['results_dir']}/best_distilled_{student_name}.pth")
        
        # Limpieza final de alumno
        del student_dist, optimizer_dist
        accelerator.free_memory()
        torch.cuda.empty_cache()

    if accelerator.is_main_process: print("\n✅ TODO LISTO. Revisa la carpeta 'results/'.")

# ✅ ENTRY POINT CORRECTO PARA 'ACCELERATE LAUNCH'
if __name__ == "__main__":
    main()
