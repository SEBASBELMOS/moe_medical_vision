# MoE Medical Vision

Implementación del proyecto de Mixture of Experts (MoE) para visión médica multimodal con 3 expertos 2D, 2 expertos 3D, backbone ViT para routing y dashboard en Streamlit.

## Estado actual resumido

### Expertos
- **Exp. 1 NIH ChestX-ray14**: ensemble híbrido NIH (ConvNeXt + especialistas por clase) — **F1 macro 0.5765**, AUC 0.7663.
- **Exp. 2 ISIC 2019**: EfficientNet-B3 — **F1 macro ~0.79**.
- **Exp. 3 Osteoarthritis**: ResNet-34 — **F1 macro ~0.84–0.90** según evaluación.
- **Exp. 4 LUNA16**: MC3-18 3D — **F1 macro 0.6571**.
- **Exp. 5 Pancreatic Cancer**: R3D-18 — **F1 macro 0.7553**.

### Router
- Ablation study formal de 4 routers rehecho en sandbox balanceado (`/workspace/router_moe_rebuild`).
- Ganador del ablation: **ViT + Linear**.
- Para runtime, la integración operativa evolucionó a un **router jerárquico**:
  - 2D color → ISIC
  - 2D radiografía → sub-router NIH vs Osteo
  - 3D → sub-router LUNA vs Páncreas

### Dashboard
- Streamlit funcional en VAST.
- URL pública temporal (quick tunnel / exposición según instancia activa).
- La app consume el `moe_system.py` del proyecto y los checkpoints reales del servidor.

## Contenido del repo
- `notebooks/`: flujo principal del proyecto.
- `src/`: datasets, modelos, losses y utilidades.
- `scripts/`: entrenamiento, tests, extracción de embeddings, recuperación y utilidades del router.
- `checkpoints/`: pesos y métricas (no siempre se versionan completas en Git).
- `embeddings/`: embeddings CLS y artefactos del router.
- `app.py`: demo de inferencia en Streamlit.

## Requisitos

### Python
- Python **3.10+** recomendado.

### Dependencias
Instalación base:
```bash
git clone https://github.com/SEBASBELMOS/moe_medical_vision.git
cd moe_medical_vision
pip install -r requirements.txt
```

Si tu entorno CUDA requiere ruedas específicas de PyTorch, instala `torch`, `torchvision` y `torchaudio` compatibles con tu versión de CUDA.

## Variables útiles
```bash
export KAGGLE_USERNAME=tu_usuario
export KAGGLE_KEY=tu_api_key
export HF_TOKEN=tu_token
```

## Reproducibilidad
Semilla fija `42` usada en notebooks y scripts principales.

## Flujo recomendado
1. `notebooks/00_setup_verificacion.ipynb`
2. `notebooks/01_train_experts_2D.ipynb`
3. `notebooks/01B_train_experts_3D.ipynb`
4. `notebooks/02_backbone_y_routers.ipynb`
5. `notebooks/03_moe_system.ipynb`

## Estructura esperada
```text
/workspace/moe_medical_vision/
|-- app.py
|-- checkpoints/
|-- embeddings/
|-- notebooks/
|-- scripts/
|-- src/
`-- data/
    `-- raw/
        |-- nih/
        |-- isic/
        |-- osteoporosis/
        |-- luna16/
        `-- pancreatic/
```

## Scripts recientes importantes
### Router / integración
- `scripts/train_router_linear_param.py`
- `scripts/router_ablation_balanced.py`
- `scripts/train_hierarchical_router.py`
- `scripts/train_router_2d_v2.py`
- `scripts/train_router_3d_v4.py`
- `scripts/test_moe_app_aligned.py`

### NIH / especialistas
- `scripts/train_nih_subset_cached_param.py`
- `scripts/train_nih_specialist.py`
- `scripts/eval_hybrid_ensemble_mass_nodule.py`

## Demo
Para levantar la app localmente:
```bash
streamlit run app.py
```

## Notas
- El README describe el estado técnico más reciente, pero algunos resultados detallados están mejor documentados en los artefactos locales del equipo (`context_engineering/` y el paper IEEE).
- Para la entrega final, el router plano y el router jerárquico deben leerse junto con el análisis del ablation y del runtime.

---

Creado por [Sebastian Belalcazar](https://github.com/SEBASBELMOS) y [Manuel Gruezo](https://github.com/manuel-gruezo)
