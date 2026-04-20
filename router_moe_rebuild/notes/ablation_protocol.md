# Router Rebuild Sandbox — Protocolo

Objetivo: rehacer el ablation study de los 4 routers en un flujo aislado, sin ensuciar .

Routers a comparar:
1. ViT + Linear
2. ViT + GMM
3. ViT + Naive Bayes
4. ViT + k-NN + PCA + FAISS

Decisión metodológica:
- Se reutilizan los embeddings ya extraídos en  y .
- Para el rerun del ablation se usa un **subset balanceado/proporcional por experto** tanto en train como en val.
- Esto evita que el desbalance natural NIH>>ISIC>>Osteo>>LUNA>>Pancreas destruya la comparación.
- El balance del router final (ViT+Linear+L_aux) se reportará sobre val balanceado, mientras que la accuracy también se mostrará en val completo natural.

Artefactos esperados del sandbox:
- 
- 
- 
- 
- 
