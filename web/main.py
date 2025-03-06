import sys
import os
import io
import torch
import numpy as np
from PIL import Image
from typing import List
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from torchvision.transforms import Compose, ToTensor, Normalize
import torchvision.transforms.functional as TF

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils.utils import load_pretrained_model, load_checkpoint
from aligner.wrapper import CVLFaceAlignmentModel, ModelConfig

import warnings
warnings.filterwarnings("ignore")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conf = ModelConfig(path='../aligner/pretrained_model/model.yaml')
aligner = CVLFaceAlignmentModel(conf, path='../aligner/pretrained_model/model.pt').to(device)
aligner.eval()

def get_aligned_face(image_input):
    """
    Преобразует входной объект в изображение PIL, применяет
    трансформации и пропускает через модель выравнивания лица.
    """
    if isinstance(image_input, Image.Image):
        img = image_input.convert('RGB')
    else:
        img = Image.open(image_input).convert('RGB')

    trans = Compose([
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    input_tensor = trans(img).unsqueeze(0).to(device)
    aligned_x, _, _, _, _, _ = aligner(input_tensor)
    aligned_x = (aligned_x[0] * 0.5 + 0.5).clamp(0, 1)
    aligned_pil = TF.to_pil_image(aligned_x)
    return aligned_pil

# Загрузка модели для получения эмбеддингов
model_ = load_pretrained_model('ir_101').to(device)
model = load_checkpoint(
    "../models/pretrained/ckpt_epoch1_batch43000_acc0.9825_eer0.0165.ckpt",
    model_
).to(device)
model.eval()

app = FastAPI()

# Разрешенные источники CORS
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:8000",
    "*",
]

# Добавляем CORS middleware к приложению
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def to_input(pil_rgb_image):
    """
    Преобразует PIL-изображение в нормализованный тензор PyTorch в формате BGR.
    """
    np_img = np.array(pil_rgb_image)
    bgr_img = np_img[:, :, ::-1]
    bgr_img_norm = (bgr_img / 255.0 - 0.5) / 0.5
    tensor = torch.tensor(bgr_img_norm.transpose(2, 0, 1), dtype=torch.float32)
    return tensor.unsqueeze(0)

def get_embedding(image):
    """
    Получает эмбеддинг загруженного изображения, предварительно выравнивая его.
    Затем вычисляет эмбеддинг с помощью загруженной модели.
    """
    image_data = image.file.read()
    img = Image.open(io.BytesIO(image_data)).convert("RGB")
    try:
        aligned_img = get_aligned_face(img)
    except Exception as e:
        print(f"Ошибка выравнивания для {img}: {e}")
        return None
    if aligned_img is None:
        print(f"Выравнивание вернуло None для {img}")
        return None
    tensor_input = to_input(aligned_img).to(device)
    with torch.no_grad():
        feature, _ = model(tensor_input)
    return feature

def compute_similarity(embeddings):
    """
    Вычисляет сходство между всеми эмбеддингами, используя скалярное произведение.
    Обрезает результаты до диапазона [0, 1].
    """
    num_images = len(embeddings)
    similarity_matrix = np.zeros((num_images, num_images))

    for i in range(num_images):
        for j in range(i + 1, num_images):
            similarity = torch.sum(
                embeddings[i] * embeddings[j],
                dim=1
            ).cpu().item()
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity  # Матрица симметрична
    similarity_matrix = np.clip(similarity_matrix, 0, 1)
    return similarity_matrix

# Монтируем директорию со статическими файлами
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    """
    Возвращает главную страницу index.html.
    """
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/about", response_class=HTMLResponse)
async def read_about():
    """
    Возвращает страницу about.html.
    """
    with open("static/about.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/verify")
async def verify_files(files: List[UploadFile] = File(...)):
    """
    Конечная точка для верификации сходства лиц по загруженным изображениям.
    Возвращает матрицу сходства и список имен файлов.
    """
    num_files = len(files)
    
    if num_files < 2 or num_files > 10:
        return JSONResponse(
            content={"error": "Загружено неверное количество файлов (от 2 до 10)"},
            status_code=400
        )

    embeddings = []
    for file in files:
        embeddings.append(get_embedding(file))

    # Вычисляем сходство между эмбеддингами
    similarity_matrix = compute_similarity(embeddings)
    similarity_matrix = similarity_matrix.tolist()

    file_names = [file.filename for file in files]

    return {
        "verification_matrix": similarity_matrix,
        "file_names": file_names
    }
