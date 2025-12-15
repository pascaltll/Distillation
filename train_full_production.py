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
# ⚙️ CONFIGURACIÓN GLOBAL (MÁXIMO RENDIMIENTO)
# =============================================================================
CONFIG = {
    # Rutas
    'train_csv': 'processed_combined_teacher/combined_train.csv',
    'val_csv':   'processed_combined_teacher/combined_val.csv',
    'results_dir': 'results',
    
    # MAESTRO (Tu WavLM Entrenado)
    'teacher_path': 'wavlm_large_finetuned_v2', 
    
    # 🎓 LISTA COMPLETA DE ALUMNOS (Incluyendo VanillaCNN)
    'students': {
        'VanillaCNN': 'VanillaCNN',      # Tu red artesanal (Baseline)
        'RepVGG': 'repvgg_a0',           # SOTA Velocidad
        'MobileOne': 'mobileone_s0',     # SOTA Móvil
        'GhostNetV2': 'ghostnetv2_100'   # SOTA Ligereza
    },
    
    # Audio
    'sample_rate': 16000,
    'duration': 3.0,
    'n_mels': 64,
    'num_classes': 4,
    
    # 🔥 OPTIMIZACIÓN DE HARDWARE (Para 2x RTX 2080 Ti)
    'batch_size': 32,    # 128 x 2 GPUs = Batch 256
    'num_workers': 0,     # Carga de datos paralela
    'epochs': 20,
    'lr': 1e-4,           # LR ajustado para batch grande
    
    # Destilación
    'temp': 4.0,
    'alpha': 0.5 
}

# Crear carpeta de resultados
os.makedirs(CONFIG['results_dir'], exist_ok=True)

# =============================================================================
# 💿 DATASET
# =============================================================================
class DualInputDataset(Dataset):
    def __init__(self, csv_path, config):
        self.df = pd.read_csv(csv_path)
        self.config = config
        self.target_len = int(config['sample_rate'] * config['duration'])
        
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config['sample_rate'], n_fft=1024, hop_length=512, n_mels=config['n_mels']
        )
        self.db_transform = T.AmplitudeToDB()
        self.label_map = {'ang': 0, 'hap': 1, 'sad': 2, 'neu': 3}

    def __len__(self): return len(self.df)

    def _load_audio(self, path):
        try:
            y, _ = librosa.load(path, sr=self.config['sample_rate'])
        except:
            y = np.zeros(self.target_len)
        
        if len(y) > self.target_len: y = y[:self.target_len]
        else: y = np.pad(y, (0, self.target_len - len(y)), mode='constant')
        return torch.tensor(y).float()

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        raw_audio = self._load_audio(row['wav_path'])
        
        mel_spec = self.mel_transform(raw_audio)
        mel_spec = self.db_transform(mel_spec)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)
        mel_spec = mel_spec.unsqueeze(0) 
        
        label = self.label_map.get(row['emotion'], 3)
        return raw_audio, mel_spec, torch.tensor(label, dtype=torch.long)

# =============================================================================
# 🍦 VANILLA CNN (Tu Red Artesanal)
# =============================================================================
class VanillaCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # Entrada: [Batch, 1, 64, Time]
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

# =============================================================================
# 🧠 FACTORY DE MODELOS (Inteligente)
# =============================================================================
class StudentModel(nn.Module):
    def __init__(self, model_name, num_classes=4):
        super().__init__()
        
        if model_name == 'VanillaCNN':
            # Cargar la red artesanal
            self.backbone = VanillaCNN(num_classes)
        else:
            # Cargar red SOTA de timm
            try:
                self.backbone = timm.create_model(model_name, pretrained=True, num_classes=num_classes, in_chans=1)
            except:
                self.backbone = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
            
    def forward(self, x):
        return self.backbone(x)

# =============================================================================
# 📝 LOGGING
# =============================================================================
def log_to_csv(epoch, train_loss, val_acc, val_uar, mode, student_name):
    # Guardamos todo en results/metrics_log.csv
    filepath = os.path.join(CONFIG['results_dir'], 'metrics_log.csv')
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Student', 'Mode', 'Epoch', 'Train_Loss', 'Val_Acc', 'Val_UAR'])
        writer.writerow([student_name, mode, epoch, f"{train_loss:.4f}", f"{val_acc:.4f}", f"{val_uar:.4f}"])

# =============================================================================
# 🔥 BUCLE UNIFICADO
# =============================================================================
def run_epoch(accelerator, student, teacher, loader, optimizer, is_train, mode="Baseline"):
    student.train() if is_train else student.eval()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, disable=not accelerator.is_local_main_process, leave=False, desc=f"{mode} {'Train' if is_train else 'Val'}")
    
    for raw_audio, mel_spec, labels in pbar:
        with torch.set_grad_enabled(is_train):
            student_logits = student(mel_spec)
            
            if mode == "Baseline":
                loss = nn.CrossEntropyLoss()(student_logits, labels)
            elif mode == "Distillation":
                with torch.no_grad():
                    teacher_logits = teacher(raw_audio).logits
                
                T = CONFIG['temp']
                soft_loss = nn.KLDivLoss(reduction="batchmean")(
                    F.log_softmax(student_logits / T, dim=1),
                    F.softmax(teacher_logits / T, dim=1)
                ) * (T**2)
                hard_loss = nn.CrossEntropyLoss()(student_logits, labels)
                loss = (CONFIG['alpha'] * soft_loss) + ((1 - CONFIG['alpha']) * hard_loss)
            
            if is_train:
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                
            total_loss += loss.item()
            preds = torch.argmax(student_logits, dim=1)
            all_preds.extend(accelerator.gather(preds).cpu().numpy())
            all_labels.extend(accelerator.gather(labels).cpu().numpy())
            
    # Calcular Métricas
    from sklearn.metrics import recall_score, accuracy_score
    min_len = min(len(all_labels), len(all_preds))
    acc = accuracy_score(all_labels[:min_len], all_preds[:min_len])
    uar = recall_score(all_labels[:min_len], all_preds[:min_len], average='macro')
    
    return total_loss / len(loader), acc, uar

# =============================================================================
# 🚀 MAIN
# =============================================================================
def main():
    accelerator = Accelerator()
    if accelerator.is_main_process:
        print(f"\n🚀 Iniciando ENTRENAMIENTO COMPLETO en {accelerator.num_processes} GPUs")
        print(f"📂 Los resultados se guardarán en: {CONFIG['results_dir']}")

    # 1. Datos (Optimizados)
    train_ds = DualInputDataset(CONFIG['train_csv'], CONFIG)
    val_ds = DualInputDataset(CONFIG['val_csv'], CONFIG)
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, 
                              num_workers=CONFIG['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, 
                            num_workers=CONFIG['num_workers'], pin_memory=True)

    # 2. Cargar Maestro
    if accelerator.is_main_process: print(f"🎓 Cargando Teacher: {CONFIG['teacher_path']}")
    try:
        teacher = AutoModelForAudioClassification.from_pretrained(CONFIG['teacher_path'], num_labels=CONFIG['num_classes'])
    except:
        teacher = AutoModelForAudioClassification.from_pretrained("microsoft/wavlm-base-plus", num_labels=CONFIG['num_classes'])
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False

    # 3. Bucle por Estudiante
    for student_name, student_arch in CONFIG['students'].items():
        if accelerator.is_main_process: 
            print(f"\n{'='*60}\n🤖 MODELO: {student_name}\n{'='*60}")

        # --- FASE 1: BASELINE ---
        if accelerator.is_main_process: print(f"📉 Entrenando Baseline...")
        student = StudentModel(student_arch, CONFIG['num_classes'])
        optimizer = optim.AdamW(student.parameters(), lr=CONFIG['lr'])
        student, optimizer, train_dl, val_dl, teacher_prepped = accelerator.prepare(student, optimizer, train_loader, val_loader, teacher)
        
        best_uar = 0
        for epoch in range(CONFIG['epochs']):
            tl, ta, tu = run_epoch(accelerator, student, None, train_dl, optimizer, True, "Baseline")
            vl, va, vu = run_epoch(accelerator, student, None, val_dl, optimizer, False, "Baseline")
            
            if accelerator.is_main_process:
                print(f"[BL] Ep {epoch+1:02d} | Loss: {tl:.3f} | UAR: {vu:.1%}")
                log_to_csv(epoch, tl, va, vu, 'Baseline', student_name)
                # GUARDAR SOLO SI MEJORA
                if vu > best_uar:
                    best_uar = vu
                    save_path = os.path.join(CONFIG['results_dir'], f"best_baseline_{student_name}.pth")
                    torch.save(accelerator.unwrap_model(student).state_dict(), save_path)
                    print(f"  💾 Nuevo Récord! Modelo guardado -> {vu:.1%}")

        del student, optimizer
        accelerator.free_memory()
        torch.cuda.empty_cache()

        # --- FASE 2: DESTILACIÓN ---
        if accelerator.is_main_process: print(f"\n🧪 Entrenando Destilación (KD)...")
        student_dist = StudentModel(student_arch, CONFIG['num_classes'])
        optimizer_dist = optim.AdamW(student_dist.parameters(), lr=CONFIG['lr'])
        student_dist, optimizer_dist, train_dl, val_dl = accelerator.prepare(student_dist, optimizer_dist, train_loader, val_loader)
        
        best_uar_dist = 0
        for epoch in range(CONFIG['epochs']):
            tl, ta, tu = run_epoch(accelerator, student_dist, teacher_prepped, train_dl, optimizer_dist, True, "Distillation")
            vl, va, vu = run_epoch(accelerator, student_dist, teacher_prepped, val_dl, optimizer_dist, False, "Distillation")
            
            if accelerator.is_main_process:
                print(f"[KD] Ep {epoch+1:02d} | Loss: {tl:.3f} | UAR: {vu:.1%}")
                log_to_csv(epoch, tl, va, vu, 'Distillation', student_name)
                # GUARDAR SOLO SI MEJORA
                if vu > best_uar_dist:
                    best_uar_dist = vu
                    save_path = os.path.join(CONFIG['results_dir'], f"best_distilled_{student_name}.pth")
                    torch.save(accelerator.unwrap_model(student_dist).state_dict(), save_path)
                    print(f"  💾 Nuevo Récord! Modelo guardado -> {vu:.1%}")

        del student_dist, optimizer_dist
        accelerator.free_memory()
        torch.cuda.empty_cache()

    if accelerator.is_main_process: print("\n✅ PROCESO COMPLETADO. Revisa la carpeta 'results/'.")

if __name__ == "__main__":
    main()
