import numpy as np
import joblib
from sklearn.naive_bayes import GaussianNB
import os

EMBEDDINGS_DIR = '/workspace/moe_medical_vision/data/processed/router_embeddings'

print("Cargando embeddings para entrenar el Router Final (Naive Bayes)...")
train_data = np.load(f"{EMBEDDINGS_DIR}/Z_train.npz")
Z_train, y_train = train_data['z'], train_data['y_expert']

print("Normalizando...")
Z_train_norm = Z_train / np.linalg.norm(Z_train, axis=1, keepdims=True)

print("Entrenando GaussianNB...")
nb = GaussianNB()
nb.fit(Z_train_norm, y_train)

print("Guardando el modelo de Router...")
joblib.dump(nb, '/workspace/moe_medical_vision/checkpoints/router_naive_bayes.pkl')
print("Router Naive Bayes listo.")
