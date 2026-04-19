from pathlib import Path

p = Path('/workspace/moe_medical_vision/src/models/moe_system.py')
text = p.read_text()

# 1. Cambiamos la importación del modelo de NIH (DenseNet en lugar de ConvNeXt)
text = text.replace("from torchvision.models import convnext_tiny, efficientnet_b3, resnet34, efficientnet_b0, resnet18", "from torchvision.models import densenet121, efficientnet_b3, resnet34, efficientnet_b0, resnet18")

# 2. Reemplazamos la función de carga del Experto 1 (NIH)
old_nih_loader = """    def _load_exp1_nih(self):
        model = convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 14)
        return model"""

new_nih_loader = """    def _load_exp1_nih(self):
        model = densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 14)
        return model"""

text = text.replace(old_nih_loader, new_nih_loader)

# 3. Cambiamos la carga del Experto 2 (ISIC) para que lea el nuevo archivo fixed
text = text.replace("self.experts[1].load_state_dict(torch.load(ckpt_dir / 'expert2_isic_best.pth'", "self.experts[1].load_state_dict(torch.load(ckpt_dir / 'expert2_isic_best_fixed.pth'")

p.write_text(text)
print("MoE System Parcheado: NIH usa DenseNet-121 y ISIC usa el checkpoint balanceado.")
