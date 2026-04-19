import sys
import time
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import streamlit as st
import nibabel as nib
from PIL import Image

sys.path.insert(0, "/workspace/moe_medical_vision/src")
from models.moe_system import MoE_System

st.set_page_config(page_title="MoE Medical Vision", page_icon="🩺", layout="wide")

if "load_balance" not in st.session_state:
    st.session_state["load_balance"] = [0, 0, 0, 0, 0]

EXPERTOS_INFO = [
    {
        "nombre": "Experto 1 (Tórax 2D)",
        "arq": "ConvNeXt-Tiny",
        "dataset": "NIH ChestX-ray14",
        "clases_map": {
            0: "Atelectasia",
            1: "Cardiomegalia",
            2: "Derrame Pleural",
            3: "Infiltración",
            4: "Masa",
            5: "Nódulo",
            6: "Neumonía",
            7: "Neumotórax",
            8: "Consolidación",
            9: "Edema",
            10: "Enfisema",
            11: "Fibrosis",
            12: "Engrosamiento Pleural",
            13: "Hernia",
        },
    },
    {
        "nombre": "Experto 2 (Piel 2D)",
        "arq": "EfficientNet-B3",
        "dataset": "ISIC 2019",
        "clases_map": {
            0: "Melanoma",
            1: "Nevo Melanocítico",
            2: "Carcinoma Basocelular",
            3: "Queratosis Actínica",
            4: "Queratosis Benigna",
            5: "Dermatofibroma",
            6: "Lesión Vascular",
            7: "Carcinoma Espinocelular",
            8: "Otro / Sano",
        },
    },
    {
        "nombre": "Experto 3 (Rodilla 2D)",
        "arq": "ResNet-34",
        "dataset": "Osteoarthritis Knee",
        "clases_map": {
            0: "Normal / Sano",
            1: "Dudoso",
            2: "Osteoartritis Leve a Severa",
        },
    },
    {
        "nombre": "Experto 4 (Nódulos 3D)",
        "arq": "MC3-18",
        "dataset": "LUNA16",
        "clases_map": {0: "Sano", 1: "Nódulo Pulmonar Positivo"},
    },
    {
        "nombre": "Experto 5 (Páncreas 3D)",
        "arq": "R3D-18",
        "dataset": "Pancreatic Cancer",
        "clases_map": {0: "Sano", 1: "Tumor Pancreático"},
    },
]


@st.cache_data
def load_random_samples(n=30):
    import random
    import itertools

    PATHS = {
        "Tórax (NIH)": (
            "/workspace/moe_medical_vision/data/raw/nih/images_001/images",
            "*.png",
        ),
        "Piel (ISIC)": (
            "/workspace/moe_medical_vision/data/raw/isic/ISIC_2019_Training_Input/ISIC_2019_Training_Input",
            "*.jpg",
        ),
        "Rodilla (Osteo)": (
            "/workspace/moe_medical_vision/data/raw/osteoporosis/KLGrade/KLGrade",
            "**/*.*",
        ),
        "Nódulo (LUNA16)": (
            "/workspace/moe_medical_vision/data/processed/luna16_highres",
            "val_*.npz",
        ),
        "Tumor Páncreas (Pancreatic)": (
            "/workspace/moe_medical_vision/data/raw/pancreatic",
            "*.nii.gz",
        ),
    }

    samples = {}
    for prefix, (path, ext) in PATHS.items():
        base_dir = Path(path)
        files_iterator = base_dir.rglob(ext) if "**" in ext else base_dir.glob(ext)
        first_files = [
            f for f in itertools.islice(files_iterator, 400)
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".npz", ".mhd", ".nii"}
            or str(f).endswith('.nii.gz')
        ]
        if first_files:
            selected = random.sample(first_files, min(n, len(first_files)))
            for s in selected:
                samples[f"{prefix} - {s.name}"] = str(s)
    samples["⚠️ Imagen Ruidosa (OOD Detection)"] = (
        "/workspace/moe_medical_vision/data/raw/isic/ISIC_2019_Training_Input/ISIC_2019_Training_Input/ISIC_0000001.jpg"
    )
    return samples


SAMPLES = load_random_samples()


@st.cache_resource
def load_moe_system():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MoE_System(device=device)
    model.load_all_weights("/workspace/moe_medical_vision/checkpoints")
    model.eval()
    router_ready = all([
        model.pca is not None,
        model.knn_index is not None,
        model.knn_labels is not None,
    ])
    return model, device, router_ready


model, device, router_ready = load_moe_system()
if not router_ready:
    st.warning("Router k-NN/FAISS no disponible en este servidor. Se usará routing de respaldo para la demo.")


def process_2d_image(image_path_or_bytes):
    img = Image.open(image_path_or_bytes).convert("RGB")
    orig_size = img.size
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    img_array = (img_array - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    tensor_224 = torch.from_numpy(img_array).float().permute(2, 0, 1).unsqueeze(0)
    return tensor_224, None, f"2D ({orig_size[0]}x{orig_size[1]}x3)", img


def process_fast_npz(file_path):
    d = np.load(file_path)
    vol = d["volume"][0]
    tensor_3d = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)

    mip = torch.stack(
        [
            torch.tensor(vol.max(axis=0)),
            torch.tensor(vol.mean(axis=0)),
            torch.tensor(vol.std(axis=0)),
        ],
        dim=0,
    )
    tensor_224 = F.interpolate(
        mip.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
    )

    mid_slice = vol[32, :, :]
    mid_slice = (mid_slice - mid_slice.min()) / (
        mid_slice.max() - mid_slice.min() + 1e-8
    )
    preview_img = Image.fromarray((mid_slice * 255).astype(np.uint8)).convert("L")

    return tensor_224, tensor_3d, "3D Pre-procesado (64x64x64)", preview_img


def process_3d_volume(file_path):
    if str(file_path).endswith(".mhd"):
        import SimpleITK as sitk

        img_sitk = sitk.ReadImage(str(file_path))
        vol = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
    else:
        vol = nib.load(str(file_path)).get_fdata(dtype=np.float32)

    orig_size = vol.shape
    vol = np.clip(vol, -1000.0, 400.0)
    vol = (vol + 1000.0) / 1400.0

    tensor_3d = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)
    tensor_3d_resized = F.interpolate(
        tensor_3d, size=(64, 64, 64), mode="trilinear", align_corners=False
    )

    x = tensor_3d_resized.squeeze(1)
    mip = torch.stack([x.max(dim=1)[0], x.max(dim=2)[0], x.max(dim=3)[0]], dim=1)
    tensor_224 = F.interpolate(
        mip, size=(224, 224), mode="bilinear", align_corners=False
    )

    mid_slice = (
        vol[orig_size[0] // 2, :, :]
        if str(file_path).endswith(".mhd")
        else vol[:, :, orig_size[2] // 2]
    )
    mid_slice = (mid_slice - mid_slice.min()) / (
        mid_slice.max() - mid_slice.min() + 1e-8
    )
    preview_img = Image.fromarray((mid_slice * 255).astype(np.uint8)).convert("L")

    return (
        tensor_224,
        tensor_3d_resized,
        f"3D ({orig_size[0]}x{orig_size[1]}x{orig_size[2]})",
        preview_img,
    )



def infer_expert_fallback(sample_choice, selected_sample_path, file_name, is_3d):
    src = f"{sample_choice} {selected_sample_path or ''} {file_name}".lower()
    if 'nih' in src or 'tórax' in src or 'torax' in src:
        return 0
    if 'isic' in src or 'piel' in src:
        return 1
    if 'osteo' in src or 'rodilla' in src or 'osteoporosis' in src or 'klgrade' in src:
        return 2
    if 'luna' in src or 'nódulo' in src or 'nodulo' in src:
        return 3
    if 'pancre' in src or 'tumor páncreas' in src or 'tumor pancreas' in src:
        return 4
    if is_3d:
        return 3
    return 0


st.title("🩺 Clasificación Médica Multimodal con MoE")

st.sidebar.title("Opciones de Entrada")
input_method = st.sidebar.radio(
    "Selecciona:", ("Usar Muestra (Sample)", "Subir Imagen Propia")
)

uploaded_file = None
selected_sample_path = None
file_name = ""
sample_choice = ""

if input_method == "Subir Imagen Propia":
    uploaded_file = st.sidebar.file_uploader(
        "Sube PNG/JPEG (2D) o NIfTI/MHD (3D)",
        type=["png", "jpg", "jpeg", "nii.gz", "mhd"],
    )
    if uploaded_file:
        file_name = uploaded_file.name
else:
    sample_choice = st.sidebar.selectbox("Elige un caso clínico:", list(SAMPLES.keys()))
    selected_sample_path = SAMPLES[sample_choice]
    file_name = Path(selected_sample_path).name

col1, col2 = st.columns([1, 1])

with col1:
    st.header("1. Preprocesado Adaptativo")

    if uploaded_file is not None or selected_sample_path is not None:
        is_3d = (
            file_name.endswith(".nii.gz")
            or file_name.endswith(".mhd")
            or file_name.endswith(".npz")
        )

        try:
            start_time = time.time()
            if is_3d:
                if uploaded_file:
                    import tempfile

                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".nii.gz"
                    ) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                    tensor_224, tensor_3d, dim_orig, preview_img = process_3d_volume(
                        tmp_path
                    )
                    os.remove(tmp_path)
                else:
                    if selected_sample_path.endswith(".npz"):
                        tensor_224, tensor_3d, dim_orig, preview_img = process_fast_npz(
                            selected_sample_path
                        )
                    else:
                        tensor_224, tensor_3d, dim_orig, preview_img = (
                            process_3d_volume(selected_sample_path)
                        )
                st.image(
                    preview_img,
                    caption="Corte Medio del Volumen 3D (Contraste Ecualizado)",
                    width=300,
                )
            else:
                target = uploaded_file if uploaded_file else selected_sample_path
                tensor_224, tensor_3d, dim_orig, preview_img = process_2d_image(target)
                st.image(preview_img, caption="Imagen 2D", width=300)

            tensor_224 = tensor_224.to(device)
            if tensor_3d is not None:
                tensor_3d = tensor_3d.to(device)

            with torch.no_grad():
                if router_ready:
                    expert_output, best_idx, expert_probs = model(
                        tensor_224, x_3d=tensor_3d
                    )
                else:
                    best_idx = infer_expert_fallback(
                        sample_choice, selected_sample_path, file_name, is_3d
                    )
                    expert_probs = torch.zeros((1, 5), device=device)
                    expert_probs[0, best_idx] = 1.0
                    expert_input = tensor_224
                    if best_idx in [3, 4] and tensor_3d is not None:
                        expert_input = tensor_3d
                        if best_idx == 3 and expert_input.shape[1] == 1:
                            expert_input = expert_input.repeat(1, 3, 1, 1, 1)
                            mean = torch.tensor([0.43216, 0.394666, 0.37645], device=device).view(1, 3, 1, 1, 1)
                            std = torch.tensor([0.22803, 0.22145, 0.216989], device=device).view(1, 3, 1, 1, 1)
                            expert_input = (expert_input - mean) / std
                    expert_output = model.experts[best_idx](expert_input)
                final_probs = torch.softmax(expert_output, dim=-1)
                pred_class = torch.argmax(final_probs, dim=-1).item()
                confidence = final_probs[0, pred_class].item() * 100

            infer_time = (time.time() - start_time) * 1000

            expert_probs_np = expert_probs[0].cpu().numpy()
            entropy = -np.sum(expert_probs_np * np.log(expert_probs_np + 1e-9))
            normalized_entropy = entropy / np.log(5)

            st.session_state["load_balance"][best_idx] += 1

            st.header("2. Resultados de Inferencia")

            nombre_patologia = EXPERTOS_INFO[best_idx]["clases_map"].get(
                pred_class, f"Desconocido ({pred_class})"
            )
            st.metric(
                label="Diagnóstico Médico",
                value=f"{nombre_patologia}",
                delta=f"{confidence:.2f}% Confianza",
            )

            st.write(f"⏱ Inferencia total: **{infer_time:.2f} ms**")

            if normalized_entropy > 0.80:
                st.error(
                    f"🚨 OOD DETECTADA: Entropía del router ({normalized_entropy:.2f}) > 0.80. La imagen no pertenece al dominio médico entrenado."
                )
            else:
                st.success(
                    f"Entropía Normal del Router: {normalized_entropy:.2f} (In-Distribution)."
                )

        except Exception as e:
            import traceback

            st.error(f"Error procesando la imagen:\n{e}\n{traceback.format_exc()}")

with col2:
    if (
        uploaded_file is not None or selected_sample_path is not None
    ) and "best_idx" in locals():
        st.header("3. Router y Load Balancing")
        exp_info = EXPERTOS_INFO[best_idx]

        st.info(
            f"**Experto Activado:** {exp_info['nombre']}\n\n**Arquitectura:** {exp_info['arq']}\n\n**Dataset Origen:** {exp_info['dataset']}\n\n**Confianza del Router:** {expert_probs_np[best_idx] * 100:.2f}%"
        )

        st.subheader("Balance de Carga Acumulado")
        import plotly.express as px

        df_lb = pd.DataFrame(
            {
                "Experto": [e["nombre"].split(" ")[1] for e in EXPERTOS_INFO],
                "Activaciones": st.session_state["load_balance"],
            }
        )
        fig = px.bar(df_lb, x="Experto", y="Activaciones", text_auto=True)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.header("4. Ablation Study del Router")
try:
    df_ablation = pd.read_csv(
        "/workspace/moe_medical_vision/ablation_study_results.csv"
    )
    st.table(df_ablation)
except:
    st.warning("No se encontró el archivo del Ablation Study.")
