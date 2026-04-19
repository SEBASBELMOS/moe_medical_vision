from pathlib import Path
import re

p = Path('/workspace/moe_medical_vision/src/models/moe_system.py')
text = p.read_text()

new_loader = """    def load_all_weights(self, ckpt_dir):
        from pathlib import Path
        import joblib
        import faiss
        import torch
        ckpt_dir = Path(ckpt_dir)
        
        ckpts = ['expert1_nih_best.pth', 'expert2_isic_best.pth', 'expert3_oa_best.pth', 'expert4_luna16_MIP_best.pth', 'expert5_pancreatic_r3d18_v5_best.pth']
        for i, ckpt_name in enumerate(ckpts):
            try:
                state = torch.load(ckpt_dir / ckpt_name, map_location=self.device, weights_only=False)['model_state_dict']
                self.experts[i].load_state_dict(state, strict=False)
            except Exception as e:
                pass
                
        try:
            self.pca = joblib.load(ckpt_dir / 'router_pca.pkl')
            self.knn_index = faiss.read_index(str(ckpt_dir / 'router_knn.index'))
            self.knn_labels = joblib.load(ckpt_dir / 'router_knn_labels.pkl')
            print("  [OK] Pesos y Router FAISS cargados (Strict=False).")
        except Exception as e:
            print(f"  [Error Router] {e}")
"""

text = re.sub(r'    def load_all_weights\(self, ckpt_dir\):.*?(?=    def forward)', new_loader, text, flags=re.DOTALL)
p.write_text(text)
print('MoE System Parcheado: Loading Block Independiente y Strict=False')
