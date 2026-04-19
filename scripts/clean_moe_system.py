from pathlib import Path

p = Path('/workspace/moe_medical_vision/src/models/moe_system.py')
text = p.read_text()

# Eliminar el hack asqueroso del temperature scaling
bad_code = "        if best_expert_idx == 1:\n            expert_output = expert_output / 3.0 # Smoothing de temperatura para ISIC\n"
if bad_code in text:
    text = text.replace(bad_code, "")
    p.write_text(text)
    print("Hack de temperatura eliminado de moe_system.py")
