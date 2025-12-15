import torch
import time
from transformers import AutoModelForAudioClassification

# Configuración
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEACHER_PATH = "wavlm_large_finetuned_v2" # O usa "microsoft/wavlm-base-plus" si no tienes la carpeta a mano

print(f"⏱️ Midiendo latencia del MAESTRO (WavLM) en {DEVICE}...")

try:
    # Cargar modelo (Intentar local, si no, HuggingFace)
    try:
        model = AutoModelForAudioClassification.from_pretrained(TEACHER_PATH).to(DEVICE)
    except:
        print("⚠️ No encontré el modelo local, descargando versión base...")
        model = AutoModelForAudioClassification.from_pretrained("microsoft/wavlm-base-plus").to(DEVICE)
    
    model.eval()

    # Input simulado (3 segundos de audio a 16kHz)
    dummy_input = torch.randn(1, 16000 * 3).to(DEVICE) # WavLM toma raw audio

    # Warmup
    print("🔥 Calentando GPU...")
    with torch.no_grad():
        for _ in range(10): _ = model(dummy_input)

    # Test
    print("🚀 Midiendo...")
    start = time.time()
    with torch.no_grad():
        for _ in range(50): _ = model(dummy_input)
    
    if DEVICE.type == 'cuda': torch.cuda.synchronize()
    end = time.time()

    avg_latency = ((end - start) / 50) * 1000
    print(f"\nResultados del MAESTRO:")
    print(f"Latencia Promedio: {avg_latency:.2f} ms")
    
except Exception as e:
    print(f"Error: {e}")
