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
from sklearn.metrics import recall_score, confusion_matrix

# =============================================================================
# ⚙️ CONFIGURACIÓN V2 (COMPLETA)
# =============================================================================
CONFIG = {
    'train_csv': 'processed_combined_teacher/combined_train.csv',
    'val_csv':   'processed_combined_teacher/combined_val.csv',
    'results_dir': 'results_v2', 
    
    'teacher_path': 'wavlm_large_finetuned_v2', 
    
    # 🎓 LOS 4 JINETES (Todos los alumnos)
    'students': {
        'VanillaCNN': 'VanillaCNN',      
        'RepVGG': 'repvgg_a0',           
        'MobileOne': 'mobileone_s0',     
        'GhostNetV2': 'ghostnetv2_100'   
    },
    
    'sample_rate': 16000, 'duration': 3.0, 'n_mels': 64, 'num_classes': 4,
    
    # ⚡ HARDWARE (Ajustado para tus 2x 2080 Ti)
    'batch_size': 112,    
    'num_workers': 0,     
    
    'epochs': 25,         
    'lr': 5e-4,
    
    # 🧪 HIPERPARÁMETROS V2
    'temp': 2.0,          
    'alpha': 0.75,        
    
    # 🛡️ DATA AUGMENTATION
    'spec_augment': True  
}

os.makedirs(CONFIG['results_dir'], exist_ok=True)

# =============================================================================
# 💿 DATASET AVANZADO (SpecAugment)
# =============================================================================
class AdvancedDataset(Dataset):
    def __init__(self, csv_path, config, is_train=False):
        self.df = pd.read_csv(csv_path)
        self.config = config
        self.target_len = int(config['sample_rate'] * config['duration'])
        self.is_train = is_train
        
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config['sample_rate'], n_fft=1024, hop_length=512, n_mels=config['n_mels']
        )
        self.db_transform = T.AmplitudeToDB()
        
        self.time_masking = T.TimeMasking(time_mask_param=30)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=10)
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
        
        if self.is_train and self.config['spec_augment']:
            mel_spec = self.time_masking(mel_spec)
            mel_spec = self.freq_masking(mel_spec)
            
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)
        mel_spec = mel_spec.unsqueeze(0) 
        
        label = self.label_map.get(row['emotion'], 3)
        return raw_audio, mel_spec, torch.tensor(label, dtype=torch.long)

# =============================================================================
# 🍦 VANILLA CNN
# =============================================================================
class VanillaCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
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
# 🧠 FACTORY
# =============================================================================
class StudentModel(nn.Module):
    def __init__(self, model_name, num_classes=4):
        super().__init__()
        if model_name == 'VanillaCNN':
            self.backbone = VanillaCNN(num_classes)
        else:
            try:
                self.backbone = timm.create_model(model_name, pretrained=True, num_classes=num_classes, in_chans=1)
            except:
                self.backbone = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
            
    def forward(self, x): return self.backbone(x)

# =============================================================================
# 📝 LOGGING
# =============================================================================
def log_detailed(epoch, loss, uar, class_acc, mode, student_name):
    path = os.path.join(CONFIG['results_dir'], 'detailed_log_v2.csv')
    exists = os.path.isfile(path)
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(['Student', 'Mode', 'Epoch', 'Loss', 'UAR', 'Acc_Ang', 'Acc_Hap', 'Acc_Sad', 'Acc_Neu'])
        writer.writerow([
            student_name, mode, epoch, f"{loss:.4f}", f"{uar:.4f}",
            f"{class_acc[0]:.2f}", f"{class_acc[1]:.2f}", f"{class_acc[2]:.2f}", f"{class_acc[3]:.2f}"
        ])

# =============================================================================
# 🔥 BUCLE DE ENTRENAMIENTO
# =============================================================================
def run_epoch(accelerator, student, teacher, loader, optimizer, is_train, mode="Baseline"):
    student.train() if is_train else student.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
    pbar = tqdm(loader, disable=not accelerator.is_local_main_process, leave=False, desc=mode)
    
    for raw_audio, mel_spec, labels in pbar:
        with torch.set_grad_enabled(is_train):
            student_logits = student(mel_spec)
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
            
    uar = recall_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2,3])
    with np.errstate(divide='ignore', invalid='ignore'):
        class_acc = cm.diagonal() / cm.sum(axis=1)
        class_acc = np.nan_to_num(class_acc)
    
    return total_loss/len(loader), uar, class_acc

# =============================================================================
# 🚀 MAIN
# =============================================================================
def main():
    accelerator = Accelerator()
    if accelerator.is_main_process:
        print(f"\n🚀 STARTING V2: 4 STUDENTS + SPECAUGMENT")
        print(f"🌡️ Temp={CONFIG['temp']} | ⚖️ Alpha={CONFIG['alpha']}")

    train_ds = AdvancedDataset(CONFIG['train_csv'], CONFIG, is_train=True)
    val_ds = AdvancedDataset(CONFIG['val_csv'], CONFIG, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

    if accelerator.is_main_process: print("🎓 Loading Teacher...")
    try:
        teacher = AutoModelForAudioClassification.from_pretrained(CONFIG['teacher_path'], num_labels=4)
    except:
        teacher = AutoModelForAudioClassification.from_pretrained("microsoft/wavlm-base-plus", num_labels=4)
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False

    for student_name, arch in CONFIG['students'].items():
        if accelerator.is_main_process: print(f"\n{'='*50}\n🤖 {student_name}\n{'='*50}")

        student = StudentModel(arch, 4)
        optimizer = optim.AdamW(student.parameters(), lr=CONFIG['lr'], weight_decay=0.01)
        
        student, optimizer, train_dl, val_dl, teacher_p = accelerator.prepare(
            student, optimizer, train_loader, val_loader, teacher
        )
        
        best_uar = 0
        for epoch in range(CONFIG['epochs']):
            tl, _, _ = run_epoch(accelerator, student, teacher_p, train_dl, optimizer, True, "Dist_V2")
            vl, vu, v_cls = run_epoch(accelerator, student, teacher_p, val_dl, optimizer, False, "Dist_V2")
            
            if accelerator.is_main_process:
                print(f"Ep {epoch+1:02d} | Loss: {tl:.3f} | UAR: {vu:.1%} | [A:{v_cls[0]:.2f} H:{v_cls[1]:.2f} S:{v_cls[2]:.2f} N:{v_cls[3]:.2f}]")
                log_detailed(epoch, tl, vu, v_cls, 'Distillation_V2', student_name)
                
                if vu > best_uar:
                    best_uar = vu
                    torch.save(accelerator.unwrap_model(student).state_dict(), f"{CONFIG['results_dir']}/best_v2_{student_name}.pth")
                    print("  💾 New Best!")

        del student, optimizer
        accelerator.free_memory()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
