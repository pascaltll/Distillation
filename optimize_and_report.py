import torch
import torch.nn as nn
import os
import pandas as pd
import timm
import time

# Configuración igual al entrenamiento
CLASSES = 4

# Definir arquitecturas (mismo código que train.py)
class VanillaCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(128, CLASSES)
    def forward(self, x): return self.classifier(self.features(x).flatten(1))

class StudentModel(nn.Module):
    def __init__(self, name):
        super().__init__()
        if name == 'VanillaCNN': self.backbone = VanillaCNN()
        else: self.backbone = timm.create_model(name, pretrained=False, num_classes=CLASSES, in_chans=1)
    def forward(self, x): return self.backbone(x)

# Función para medir tamaño
def get_size_mb(model):
    torch.save(model.state_dict(), "temp.pth")
    size = os.path.getsize("temp.pth") / (1024 * 1024)
    os.remove("temp.pth")
    return size

# Función para medir latencia (simulada en CPU)
def benchmark(model, input_shape=(1, 1, 64, 200)):
    model.eval()
    x = torch.randn(input_shape)
    # Warmup
    for _ in range(10): model(x)
    # Test
    start = time.time()
    for _ in range(100): model(x)
    end = time.time()
    return (end - start) / 100 * 1000 # ms

# --- MAIN ---
results_dir = "results"
models_to_test = {
    'VanillaCNN': 'VanillaCNN',
    'RepVGG': 'repvgg_a0',
    'MobileOne': 'mobileone_s0',
    'GhostNetV2': 'ghostnetv2_100'
}

print(f"{'Model':<15} | {'Original (MB)':<15} | {'Int8 (MB)':<15} | {'Reduction':<10} | {'Latency (ms)':<15}")
print("-" * 80)

for name, arch in models_to_test.items():
    # 1. Cargar FP32 (Original)
    model_fp32 = StudentModel(arch)
    path = f"{results_dir}/best_distilled_{name}.pth"
    
    try:
        # Cargar pesos (ajustando keys si vienen de DDP/Accelerate)
        state_dict = torch.load(path, map_location='cpu')
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model_fp32.load_state_dict(new_state_dict)
    except:
        print(f"⚠️ No se encontró {path}, saltando...")
        continue

    size_fp32 = get_size_mb(model_fp32)
    lat_fp32 = benchmark(model_fp32)

    # 2. Cuantizar a INT8 (Dynamic)
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )
    size_int8 = get_size_mb(model_int8)
    
    reduction = size_fp32 / size_int8
    
    print(f"{name:<15} | {size_fp32:.2f} MB        | {size_int8:.2f} MB        | {reduction:.1f}x        | {lat_fp32:.2f} ms")

print("-" * 80)
