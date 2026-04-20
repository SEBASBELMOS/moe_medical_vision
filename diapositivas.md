---
marp: true
theme: default
paginate: true
size: 16:9
header: "UAO — Incorporar Elementos de IA · Bloque Visión"
footer: "Belalcazar · Gruezo — Abril 2026"
style: |
  section { font-family: "Times New Roman", serif; font-size: 22px; }
  h1 { color: #1f3a5f; }
  h2 { color: #1f3a5f; }
  table { font-size: 18px; }
  code { font-size: 18px; }
---

<!-- _class: lead -->

# Clasificación Médica Multimodal con **Mixture of Experts**
## Ablation Study del Router

**Sebastián Belalcazar Mosquera** · **Manuel Alejandro Gruezo**
Docente: Carlos Andrés Ferro Sánchez
Universidad Autónoma de Occidente — Abril 2026

<!--
Slide 1 (~45 s). Saludo, presentación del equipo y del tema. Aclarar que el trabajo
tiene dos ejes: construir el sistema MoE funcional y responder una pregunta científica.
-->

---

## 1. El problema y la pregunta científica

- Los sistemas clínicos reales procesan **múltiples modalidades** (rayos X, dermoscopía, CT) sin saber de antemano qué reciben.
- Un modelo por modalidad no escala; un modelo generalista pierde especialización.
- **MoE** resuelve el dilema: un *router* activa dinámicamente al experto adecuado.

> **Pregunta científica central**
> ¿Justifica el Vision Transformer su costo computacional como router frente a métodos estadísticos clásicos operando sobre los mismos embeddings?

<!--
Slide 2 (~60 s). Enfatizar que el router no es un detalle — es el objeto de estudio.
La respuesta hay que demostrarla con datos, no asumirla.
-->

---

## 2. Datasets — 5 modalidades, sin metadatos

| # | Dataset | Modalidad | Clases | Volumen |
|---|---------|-----------|--------|---------|
| 1 | NIH ChestX-ray14 | Rayos X 2D | 5 patol. | ~112 K img |
| 2 | ISIC 2019 | Dermoscopía 2D | 9 | ~25 K img |
| 3 | Osteoarthritis | RX rodilla 2D | 3 | ~10 K img |
| 4 | LUNA16 | CT pulmonar 3D | 2 | 888 vol. |
| 5 | Pancreatic Cancer | CT abdominal 3D | 2 | ~281 vol. |

**Restricción:** el sistema recibe **solo la imagen** — nada de texto, ni modalidad, ni nombre de archivo. Soluciones que pasen metadatos pierden 20%.

<!--
Slide 3 (~60 s). Subrayar la heterogeneidad (2D vs 3D, prevalencias, clases).
Mencionar que el desbalance natural (NIH ≫ Pancreas) sesga cualquier evaluación.
-->

---

## 3. Preprocesador adaptativo — sin metadatos

Detección **estructural** por rango del tensor:

```
rank(x) == 4  →  imagen 2D  →  Resize 224×224 + normalización ImageNet
rank(x) == 5  →  volumen 3D →  Resize 64×64×64 + HU [-1000, 400]
```

- Sin `if modality == "xray"`: la modalidad **se infiere**, no se recibe.
- Elimina por diseño la penalización de -20% por uso de metadatos.
- Permite que una misma interfaz sirva para PNG/JPEG/NIfTI.

<!--
Slide 4 (~45 s). Este es un punto fuerte del proyecto: el adaptive preprocessor es
lo que hace que todo el pipeline sea honesto respecto a la restricción del enunciado.
-->

---

## 4. Arquitectura del sistema MoE

```
Imagen → AdaptivePreprocessor → ViT-Tiny (frozen) → z ∈ R¹⁹²
                                                      │
                                                      ▼
                                              Router A (965 params)
                                                      │  argmax p
                                                      ▼
                         ┌───────────┬───────────┬───────────┐
                      E1 NIH      E2 ISIC      E3 OA      E4 LUNA    E5 Panc
                    ConvNeXt  EffNet-B3  ResNet-34  MIP+EffN-B0  R3D-18
```

- Método **B (por partes)**: expertos entrenados individualmente → congelados → router aprende sobre embeddings.
- Backbone compartido y congelado ⇒ el ablation es **justo** (todos los routers compiten sobre el *mismo* $z$).
- Expertos **heterogéneos** (requisito del proyecto).

<!--
Slide 5 (~75 s). Explicar el flujo. Insistir en "mismo embedding para los 4 routers"
— es lo que vuelve el experimento comparable.
-->

---

## 5. Ablation Study — los 4 mecanismos

| Router | Naturaleza | Ecuación clave |
|--------|------------|----------------|
| **A — ViT + Linear** | DL (gradiente) | $g(z)=\text{softmax}(Wz+b)$ |
| **B — GMM** | Paramétrico (EM) | $P(e_i\|z) \propto \pi_i\,\mathcal{N}(z;\mu_i,\Sigma_i)$ |
| **C — Naive Bayes** | Paramétrico (MLE, analítico) | $\prod_j \mathcal{N}(z_j;\mu_{ij},\sigma^2_{ij})$ |
| **D — k-NN + PCA** | No paramétrico | $\arg\max_i \sum_{j\in \text{kNN}} \mathbb{1}[e_j=i]$ |

- PCA 192→32 para D (varianza explicada 84.5%) — mitiga maldición de dimensionalidad.
- Embeddings extraídos una sola vez: $Z_\text{train}\in\mathbb{R}^{121287\times192}$.

<!--
Slide 6 (~75 s). Resaltar que son 4 naturalezas matemáticas distintas: un gradiente,
dos paramétricos estadísticos (uno iterativo, uno analítico) y uno no paramétrico.
-->

---

## 6. Evaluación justa — subset balanceado

**Problema:** el val natural está dominado por NIH (77%). Un router tonto que siempre diga "NIH" obtendría ~77% de accuracy → métrica engañosa.

**Solución metodológica:**
- Muestreo estratificado: 1 000 por clase en train (5 000 total), 200 por clase en val balanceado.
- Se reportan **dos métricas**:
  - **Acc. (bal)** → capacidad discriminativa pura (la métrica justa para comparar)
  - **Acc. (full)** → comportamiento operativo bajo la distribución real

La brecha entre ambas es un **diagnóstico**: mide cuánto depende el router de la prior empírica.

<!--
Slide 7 (~60 s). Punto metodológico importante. Sin esto, GMM y NB se ven mucho peor
de lo que son. El jurado aprecia que seamos explícitos con esto.
-->

---

## 7. Resultados del ablation study

| Router | Acc. **bal** | Acc. **full** | Latencia | Params | GPU |
|--------|:---:|:---:|:---:|:---:|:---:|
| **A — ViT + Linear** | **99.60 %** | **98.27 %** | 0.000 ms | 965 | Sí |
| D — k-NN + PCA | 99.20 % | 97.11 % | 1.447 ms | 160 000 | No |
| B — GMM | 95.80 % | 87.35 % | 0.004 ms | 1 920 | No |
| C — Naive Bayes | 95.80 % | 87.34 % | 0.002 ms | 1 920 | No |

- **Los 4 superan la meta de 80 %** en los dos regímenes.
- Router A gana; k-NN está a menos de 1.2 pp en ambos regímenes.
- GMM y NB son prácticamente indistinguibles entre sí.

<!--
Slide 8 (~75 s). Slide central del proyecto. Leer los números en voz alta. Hacer
notar la convergencia: el ablation responde la pregunta científica con datos.
-->

---

## 8. Discusión — ¿qué nos dicen los números?

**Hipótesis natural (ViT gana):** ✅ confirmada — pero **matizada**.
- Diferencia A vs k-NN: **0.40 pp** (bal) / **1.16 pp** (full) → prácticamente empate.
- Diferencia A vs NB: **3.8 pp** (bal) — sorprendentemente poco.

**Lección principal:**
> La **calidad del embedding** importa más que la complejidad del routing.
> Un ViT bien preentrenado genera representaciones donde hasta Naive Bayes pasa el 95% balanceado.

**Implicación práctica:** en hardware restringido, **k-NN+PCA** o **Naive Bayes** son alternativas viables (sin GPU, 0.002 ms).

<!--
Slide 9 (~75 s). Aquí está la contribución científica. No decir "ViT ganó, fin" —
decir "ViT gana, pero la diferencia es pequeña porque el embedding ya resuelve el problema".
-->

---

## 9. Balance de carga — Auxiliary Loss

$$\mathcal{L}_\text{aux} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i \qquad (\alpha = 0.20)$$

- **Expert Collapse** (router siempre al mismo experto) → destruye la especialización.
- **Rúbrica:** $\max(f_i)/\min(f_i) < 1.30$ o se pierde el 40 % de la nota.
- Sweep de $\alpha \in \{0.05, 0.10, 0.20\}$ → $\alpha = 0.20$ evita ratios transitorios $> 1.30$.

**Resultado final:** ratio $= \mathbf{1.18} < 1.30$ ✅ (margen cómodo, sin colapso).

<!--
Slide 10 (~60 s). Es el punto de mayor riesgo en la rúbrica. Decir claramente que
cumplimos con margen. El α=0.2 es el resultado de un sweep real.
-->

---

## 10. Expertos heterogéneos — resultados

| Expert | Dataset | Arquitectura | F1 Macro |
|:---:|---------|--------------|:---:|
| 1 | NIH ChestX-ray14 | **ConvNeXt ensemble + especialistas Mass/Nodule** | 0.577 |
| 2 | ISIC 2019 | EfficientNet-B3 | 0.792 |
| 3 | Osteoarthritis | ResNet-34 | **0.836** |
| 4 | LUNA16 | MIP + EfficientNet-B0 | 0.615 |
| 5 | Pancreatic | R3D-18 + FocalLoss | 0.755 |

- **Experto 1 (NIH):** migramos de DenseNet-121 multi-label (F1=0.327) a **ConvNeXt + 2 especialistas + umbral re-tunado** → F1=0.577 (**+25 pp**).
- Meta 2D > 0.72 ✅ (E2, E3). Meta 3D > 0.65 ✅ (E5). E1 y E4 por debajo → clases de baja prevalencia.

<!--
Slide 11 (~75 s). Destacar el salto de +25 pp en NIH como contribución de ingeniería
(no solo "pusimos un modelo más grande"). Ser honestos con lo que no alcanzó la meta.
-->

---

## 11. Evaluación end-to-end y OOD

| Dominio | N_val | Route Acc. | F1 Expert | F1 E2E |
|---------|:---:|:---:|:---:|:---:|
| NIH | 16 818 | 0.9934 | 0.577 | 0.573 |
| ISIC | 3 799 | 0.9958 | 0.792 | 0.789 |
| Osteoarthritis | 953 | 0.9990 | 0.836 | 0.835 |
| LUNA16 MIP | 177 | 1.0000 | 0.615 | 0.615 |
| | | | **Macro** | **0.7030** |

**Calibración del router como detector OOD:**
- AUROC = 0.9845 sin entrenamiento adicional.
- $\tau = 0.95$: cubre 96.8% con acc 0.9994 → *reject-option* gratis.

<!--
Slide 12 (~60 s). Mostrar que el sistema integrado funciona y que la confianza del
router es útil más allá del routing (detección OOD). Esto cierra el requisito de rúbrica.
-->

---

## 12. Dashboard interactivo

**Funcionalidades obligatorias implementadas:**
- 📂 Carga PNG / JPEG / NIfTI con detección automática 2D/3D.
- ⏱️ Inferencia en tiempo real: etiqueta, confianza, latencia.
- 🔥 **Attention heatmap** del router ViT (rollout Abnar & Zuidema).
- 🧭 Panel del experto activado + gating score.
- 📊 Tabla del ablation study (los 4 routers con métricas reales).
- ⚖️ Barras de load balance ($f_i$ acumulado).
- 🚨 Alerta OOD cuando la entropía del gating supera el umbral.
- 🔀 Selector Router A / Router D (k-NN) para demo comparativa.

<!--
Slide 13 (~60 s). Mencionar que el dashboard es la demo en vivo. Mostrar (si hay
proyector) o prometer demo al final. El selector A/D es valor añadido que no pide
la rúbrica pero refuerza el ablation.
-->

---

## 13. Alcances y limitaciones

**Alcances ✅**
- Preprocesador adaptativo sin metadatos (5 modalidades).
- Los 4 routers superan la meta de 80 % — pregunta científica respondida.
- Balance de carga dentro del umbral (ratio 1.18 < 1.30).
- Dashboard con heatmap, OOD y selector de routers.

**Limitaciones ⚠️**
- Expertos 3D entrenados sobre subsets (LUNA 888 vol., Pancreas ~558 vol.).
- Router LUNA usa un **proxy MIP** (max/mean/std), no el volumen 3D real → el heatmap localiza "pulmón", no el nódulo.
- NIH entrenado sobre 5 patologías (no 14); Mass y Nodule siguen arrastrando el F1 macro.

**Con más tiempo:** backbone 3D médico (MedicalNet), datasets completos, Top-K gating.

<!--
Slide 14 (~60 s). Ser honestos. La rúbrica valora explícitamente admitir qué no
funcionó. El comentario sobre el heatmap de LUNA anticipa una pregunta común del jurado.
-->

---

<!-- _class: lead -->

## 14. Conclusiones

- **La pregunta científica queda respondida con datos:** el ViT justifica su costo *en precisión*, pero por un margen pequeño.
- **La calidad del embedding > la complejidad del routing**: con un ViT-Tiny preentrenado, hasta Naive Bayes supera el 95 % balanceado.
- Sistema MoE integrado: **99.60 % Routing Acc. balanceada**, **F1 macro E2E = 0.7030**, ratio de balance **1.18**.
- El proyecto demuestra que el aporte de valor está en el **diseño experimental honesto**, no en forzar que el ViT "gane" por mucho.

### Gracias — ¿preguntas?

<!--
Slide 15 (~45 s). Cierre. Repetir la lección principal. Abrir a Q&A. Si queda tiempo,
ofrecer demo en vivo del dashboard.
-->
