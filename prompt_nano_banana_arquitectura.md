# Prompt para Nano Banana — Diagrama de Arquitectura MoE

> Modelo sugerido: `gemini-2.5-flash-image` (Nano Banana). Aspect ratio 16:9 o 4:3. Si la primera salida sale con texto borroso, pedir "re-render with crisp typography, no blur" y reenviar.

---

## Prompt principal (copiar/pegar)

```
Create a clean, professional technical architecture diagram of a Mixture-of-Experts
(MoE) medical imaging system. Flat vector style, white background, thin dark-gray
strokes, rounded rectangular boxes with soft pastel fills, crisp readable
sans-serif typography (Inter or similar), no 3D effects, no drop shadows, no
gradients. Publication-ready for an academic technical report.

Layout: left-to-right horizontal data flow, with the 5 experts stacked vertically
on the right side. The canvas should feel balanced, like an IEEE paper figure.

PIPELINE — draw the following 5 stages connected by arrows with solid arrowheads:

Stage 1 — INPUT (light gray fill):
  Title: "Entrada"
  Body: "Imagen o Volumen médico"
  Subtext: "PNG / JPEG / NIfTI — sin metadatos"
  Tensor shape note: "2D: [B, 3, H, W]   ó   3D: [B, 1, D, H, W]"

Stage 2 — ADAPTIVE PREPROCESSOR (light blue fill):
  Title: "Adaptive Preprocessor"
  Body two lines:
    "rank(x)=4 → Resize 224×224 + ImageNet norm"
    "rank(x)=5 → Resize 64×64×64 + HU[-1000,400]"
  Small tag below: "detección estructural, sin metadatos"

Stage 3 — BACKBONE (light cyan fill, with a small snowflake/lock icon to mean
"frozen"):
  Title: "ViT-Tiny (frozen)"
  Body: "Patch Embedding + Self-Attention"
  Subtext: "ImageNet-21k preentrenado"
  Output arrow labeled: "z ∈ ℝ¹⁹²  ([CLS] token)"

Stage 4 — ROUTER ABLATION BRANCH (orange/amber theme):
  Draw a horizontal dashed bracket around FOUR parallel router boxes, labeled
  above the bracket: "ABLATION STUDY — 4 mecanismos sobre el mismo z"

  The four router boxes (same size, aligned in a row, amber fill, with a small
  medal icon on Router A to indicate "ganador"):

  Router A — "ViT + Linear + Softmax"
    subtitle: "DL (gradiente) · 965 params"
    metric badge: "Acc bal 99.60%  ✓ ganador"

  Router B — "ViT + GMM"
    subtitle: "Paramétrico (EM) · 1920 params"
    metric badge: "Acc bal 95.80%"

  Router C — "ViT + Naive Bayes"
    subtitle: "Paramétrico (MLE) · 1920 params"
    metric badge: "Acc bal 95.80%"

  Router D — "ViT + k-NN + PCA + FAISS"
    subtitle: "No paramétrico · 160k valores"
    metric badge: "Acc bal 99.20%"

  From Router A (the winner) draw a thick solid arrow going right, labeled
  "argmax p ∈ ℝ⁵". From the other three routers draw thin, faded arrows that
  stop before reaching the experts (they do not continue in production).

Stage 5 — EXPERTS PANEL (green theme, light green fills, grouped inside a
rounded container labeled on top: "5 Expertos heterogéneos (pesos congelados)"):

  Stack 5 boxes vertically:
    E1 — NIH ChestX-ray14       · ConvNeXt ensemble + specs.   · 2D · F1 0.577
    E2 — ISIC 2019              · EfficientNet-B3              · 2D · F1 0.792
    E3 — Osteoarthritis         · ResNet-34                     · 2D · F1 0.836
    E4 — LUNA16                 · MIP + EfficientNet-B0         · 3D · F1 0.615
    E5 — Pancreatic Cancer      · R3D-18 + FocalLoss            · 3D · F1 0.755

  The arrow from Router A fans out to all 5 experts with thin gray lines. One
  line (to the selected expert, any of them for illustration — use E3) is drawn
  thick and black to represent the argmax selection. The others remain gray.

Stage 6 — OUTPUT (light red/coral fill on the far right):
  Title: "ŷ"
  Subtitle: "Predicción clínica"

AUXILIARY LOSS OVERLAY:
  Above Router A, draw a dashed orange rounded rectangle labeled:
  "L_aux = α · N · Σ fᵢ · Pᵢ    (α = 0.20, Switch Transformer)"
  with a dashed orange arrow curving down into Router A. Small caption:
  "ratio max/min = 1.18 < 1.30 ✓ (sin Expert Collapse)"

ANNOTATIONS / LEGEND (bottom of the figure, small font):
  Three color chips with labels:
    ■ cyan  = componente congelado (frozen, no grad)
    ■ amber = router (evaluado en ablation)
    ■ green = experto (entrenado individualmente)

  Small footnote: "Sin metadatos externos. Detección 2D/3D estructural por rank
  del tensor."

TYPOGRAPHY:
  - Section titles: bold, 14–16 pt equivalent
  - Body labels: regular, 11–12 pt
  - Metric badges: bold, 10 pt, inside a pill shape
  - All text must be perfectly legible, no typos, no hallucinated extra labels.

DO NOT include:
  - photorealistic medical imagery
  - stock icons unrelated to the diagram
  - watermarks or branding
  - color gradients or glow effects
  - extra decorative boxes not described above

The overall aesthetic should read like a figure from a Springer/IEEE paper:
minimalist, informative, every element earns its place.
```

---

## Variante corta (si nano banana corta texto largo)

Si el modelo devuelve un diagrama con texto truncado o borroso, usa esta versión condensada:

```
Professional flat technical diagram, IEEE paper style, white background, thin
gray strokes, pastel fills, sans-serif crisp text. Left-to-right flow:

[1] INPUT (gray) "Imagen/Volumen médico · sin metadatos · 2D o 3D"
  →
[2] ADAPTIVE PREPROCESSOR (blue) "rank=4 → 224×224 | rank=5 → 64³"
  →
[3] VIT-TINY FROZEN (cyan, snowflake icon) "ImageNet-21k · [CLS] z∈ℝ¹⁹²"
  →
[4] ABLATION: 4 routers in parallel (amber), bracketed as "Ablation Study":
    A) ViT+Linear (WINNER, medal) — 99.60% bal
    B) ViT+GMM — 95.80% bal
    C) ViT+Naive Bayes — 95.80% bal
    D) ViT+k-NN+PCA+FAISS — 99.20% bal
  Thick arrow only from A →
[5] 5 EXPERTS stacked (green): E1 NIH ConvNeXt · E2 ISIC EffNet-B3 ·
    E3 OA ResNet-34 · E4 LUNA MIP+EffNetB0 · E5 Pancreas R3D-18.
    Thin arrows to all, one thick to E3 (argmax).
  →
[6] OUTPUT (coral) "ŷ"

Overlay above Router A: dashed orange box "L_aux = α·N·Σ fᵢPᵢ, α=0.20,
ratio 1.18 < 1.30 ✓". Dashed arrow into Router A.

Legend bottom: cyan=frozen, amber=router, green=expert. Footnote: "sin
metadatos, detección 2D/3D por rank del tensor". No 3D effects, no gradients,
no shadows.
```

---

## Prompt alternativo en inglés (algunos modelos rinden mejor)

```
Clean flat technical architecture diagram for a Mixture-of-Experts medical
imaging system. IEEE paper style. White background, rounded boxes, pastel
fills, thin gray strokes, crisp sans-serif text, no gradients, no shadows.

Left-to-right pipeline:
Input (gray): "Medical image or volume · no metadata · 2D or 3D tensor"
→ Adaptive Preprocessor (blue): "rank-4 → 224×224 ImageNet norm | rank-5 → 64³ HU"
→ ViT-Tiny backbone, frozen (cyan, with small lock icon): "[CLS] z ∈ ℝ¹⁹²"
→ Ablation bracket "4 routers over the same z" containing in parallel:
    • ViT + Linear — winner medal — 99.60% bal acc
    • ViT + GMM — 95.80%
    • ViT + Naive Bayes — 95.80%
    • ViT + k-NN + PCA + FAISS — 99.20%
  Only Router A continues with a thick arrow labeled "argmax p".
→ Group of 5 heterogeneous experts (green), frozen weights:
    E1 NIH ConvNeXt ensemble · E2 ISIC EfficientNet-B3 · E3 Osteoarthritis
    ResNet-34 · E4 LUNA16 MIP+EffNetB0 · E5 Pancreas R3D-18.
  One expert highlighted (E3) receives the thick arrow (argmax selection).
→ Output ŷ (coral).

Overlay an orange dashed box above Router A labeled
"L_aux = α·N·Σ fᵢ·Pᵢ  (α=0.20, ratio 1.18 < 1.30)" with a dashed arrow feeding
back into Router A.

Legend at bottom: cyan = frozen, amber = router, green = expert. Footnote:
"No external metadata; 2D/3D detected by tensor rank." Keep all labels in
Spanish where marked. No photorealistic imagery, no extra decoration.
```

---

## Tips de iteración

1. **Si los labels se mezclan:** pídele "re-render with larger font size and more whitespace between boxes".
2. **Si los números salen mal:** adjunta una imagen de referencia (el diagrama TikZ del `.tex`) y dile "keep this exact layout and textual content".
3. **Si quieres vertical en lugar de horizontal** (para slide portrait): reemplaza "left-to-right flow" por "top-to-bottom flow" en el prompt.
4. **Exportar:** pedir explícitamente "export as PNG at 300 DPI, 1920×1080 px".
5. **Para el reporte LaTeX:** guardar en `POD/moe_medical_vision/checkpoints/figures/fig1_architecture.pdf` y reemplaza el bloque TikZ actual por `\includegraphics[width=\linewidth]{fig1_architecture.pdf}`.
