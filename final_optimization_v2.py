import os
import torch
import torch.nn as nn
import timm
import time
import copy
import numpy as np
import torch.quantization # Necesario para las funciones de cuantización

# =============================================================================
# ⚙️ CONFIGURACIÓN
# =============================================================================
CLASSES = 4
RESULTS_DIR = "results_v2" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_MAP = {
    'VanillaCNN': 'VanillaCNN',
    'RepVGG': 'repvgg_a0',
    'MobileOne': 'mobileone_s0',
    'GhostNetV2': 'ghostnetv2_100'
}

print(f"🚀 Protocolo de Optimización iniciado en: {DEVICE}")
print("-" * 105) 

# =============================================================================
# 1. DEFINICIÓN DE ARQUITECTURAS (INCLUIDAS DE NUEVO)
# =============================================================================
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
        # Stubs para cuantización estática
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

class StudentModel(nn.Module):
    def __init__(self, name):
        super().__init__()
        if name == 'VanillaCNN':
            self.backbone = VanillaCNN()
        else:
            try:
                # timm models are loaded here
                self.backbone = timm.create_model(name, pretrained=False, num_classes=CLASSES, in_chans=1)
            except:
                self.backbone = timm.create_model(name, pretrained=False, num_classes=CLASSES)

    def forward(self, x): return self.backbone(x)

# =============================================================================
# 2. HERRAMIENTAS DE MEDICIÓN
# =============================================================================
def get_size_mb(model):
    torch.save(model.state_dict(), "temp.pth")
    size = os.path.getsize("temp.pth") / (1024 * 1024)
    if os.path.exists("temp.pth"): os.remove("temp.pth")
    return size

def measure_latency(model, input_shape=(1, 1, 64, 200), device='cpu', precision='fp32'):
    """Mide la latencia promedio en ms"""
    model.to(device)
    model.eval()

    # Crear input dummy
    dtype = torch.float16 if precision == 'fp16' else torch.float32
    x = torch.randn(input_shape, dtype=dtype).to(device)

    # Warmup (calentar la GPU/CPU)
    with torch.no_grad():
        for _ in range(10): _ = model(x)

    # Test Loop
    start = time.time()
    with torch.no_grad():
        for _ in range(100): _ = model(x)

    if device.type == 'cuda': torch.cuda.synchronize()
    end = time.time()

    return ((end - start) / 100) * 1000 # ms

def apply_static_quantization(model):
    """Calibración y conversión a INT8 (CPU)"""
    model.eval()
    model.cpu()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)

    # Calibración (Simulamos datos)
    with torch.no_grad():
        for _ in range(20):
            dummy = torch.randn(1, 1, 64, 200)
            model(dummy)

    torch.quantization.convert(model, inplace=True)
    return model

# =============================================================================
# 3. EJECUCIÓN PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    
    print(f"{'Model':<12} | {'Orig (MB)':<10} | {'Opt (MB)':<10} | {'Method':<8} | {'Reduct':<8} | {'Lat FP32 (ms)':<15} | {'Lat OPT (ms)':<15}")
    print("-" * 105)

    for name, arch in MODELS_MAP.items():
        # 1. Cargar FP32
        path = f"{RESULTS_DIR}/best_v2_{name}.pth"
        if not os.path.exists(path):
            print(f"⚠️ {name}: No encontrado en {path}")
            continue

        model = StudentModel(arch)
        try:
            state_dict = torch.load(path, map_location='cpu')
            
            # Limpieza de claves y carga
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace('module.', '').replace('backbone.', '')
                new_state_dict[k] = v
            
            # Carga flexible
            msg = model.load_state_dict(new_state_dict, strict=False)
            if name == 'VanillaCNN' and len(msg.missing_keys) > 0:
                 model.backbone.load_state_dict(new_state_dict, strict=False)

        except Exception as e:
            print(f"⚠️ Error cargando {name}: {e}")
            continue

        # 📏 MEDICIÓN DEL ESTADO ORIGINAL (FP32)
        size_orig = get_size_mb(model)
        
        # Medir latencia original (usamos CPU para Vanilla, GPU para otros)
        if name == 'VanillaCNN':
            lat_orig = measure_latency(model, device=torch.device('cpu'), precision='fp32')
        else:
            lat_orig = measure_latency(model, device=DEVICE, precision='fp32') 

        # --- ESTRATEGIA DE OPTIMIZACIÓN ---

        if name == 'VanillaCNN':
            method = "INT8"
            try:
                model_opt = copy.deepcopy(model.backbone)
                model_opt = apply_static_quantization(model_opt)
                size_opt = get_size_mb(model_opt)
                lat_opt = measure_latency(model_opt, device=torch.device('cpu')) 
            except Exception as e:
                method = "ERR"
                size_opt = size_orig
                lat_opt = 0.0 # Asegura float
                
        else:
            method = "FP16"
            model_opt = copy.deepcopy(model).half()
            size_opt = get_size_mb(model_opt)
            lat_opt = measure_latency(model_opt, device=DEVICE, precision='fp16')

        # Calcular Métricas Finales
        reduction = size_orig / size_opt

        print(f"{name:<12} | {size_orig:<10.2f} | {size_opt:<10.2f} | {method:<8} | {reduction:<8.1f}x | {lat_orig:<15.2f} | {lat_opt:<15.2f}")

    print("-" * 105)
    print("✅ Optimización y comparación de latencia completada.")
