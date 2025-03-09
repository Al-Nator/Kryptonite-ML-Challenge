#!/usr/bin/env python3
"""
Модуль для вычисления cosine similarity матриц для датасета.
Принимает аргументы для путей к данным и сохраняет результаты в NPZ-файл.
В этом варианте обрабатываются все изображения, для которых существует запись в meta.json.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
from tqdm import tqdm

# Предполагается, что модуль net доступен в PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.utils.utils import load_pretrained_model

# -------------------------------
# Параметры устройства и трансформация для инференса
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inference_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def compute_cosine_similarity(base_dir, meta, model, transform, device):
    """
    Вычисляет матрицы cosine similarity для каждого человека в датасете.

    Для каждой папки (человека) перебираются файлы, и выбираются только те, для которых
    в meta.json существует запись (независимо от значения). Для выбранных файлов вычисляются
    эмбеддинги, нормализуются и вычисляется матрица cosine similarity.

    Args:
        base_dir (str): Путь к датасету изображений (структура: base_dir/person/filename).
        meta (dict): Словарь meta.json, где ключи вида "person/filename".
        model: Загруженная модель AdaFace.
        transform: Трансформация для инференса.
        device: Устройство (cuda/cpu).

    Returns:
        dict: Результаты, где ключ – имя человека (папка), а значение – словарь с ключами:
              'cosine_matrix' и 'file_names'.
    """
    results = {}
    person_folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    for person in tqdm(person_folders, desc="Processing persons"):
        person_dir = os.path.join(base_dir, person)
        valid_files = []
        # Для каждого файла проверяем, что запись существует в meta.json
        for filename in os.listdir(person_dir):
            key = f"{person}/{filename}"
            if key in meta:
                valid_files.append(filename)
        if not valid_files:
            continue
        valid_files.sort()  # Для консистентного порядка
        images = []
        for file in valid_files:
            img_path = os.path.join(person_dir, file)
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Ошибка загрузки изображения {img_path}: {e}")
                continue
            img_tensor = transform(img)
            images.append(img_tensor)
        if not images:
            continue
        images_tensor = torch.stack(images).to(device)
        with torch.no_grad():
            embeddings, _ = model(images_tensor)
            embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
            cosine_matrix = torch.matmul(embeddings, embeddings.t()).cpu().numpy()
        results[person] = {
            'cosine_matrix': cosine_matrix,
            'file_names': valid_files
        }
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Модуль для вычисления cosine similarity матриц для датасета."
    )
    parser.add_argument("--base_dir", type=str, default="data/train/images_aligned",
                        help="Путь к датасету изображений.")
    parser.add_argument("--meta_path", type=str, default="data/train/meta.json",
                        help="Путь к файлу meta.json")
    parser.add_argument("--output_file", type=str, default="data/cosine_similarity_results.npz",
                        help="Путь для сохранения NPZ-файла с результатами")
    parser.add_argument("--architecture", type=str, default="ir_101",
                        help="Архитектура модели AdaFace.")
    args = parser.parse_args()

    # Загрузка модели
    model = load_pretrained_model(args.architecture, load_weights=True).to(device)
    model.eval()

    # Загрузка meta.json
    with open(args.meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Вычисление cosine similarity
    results = compute_cosine_similarity(args.base_dir, meta, model, inference_transform, device)

    # Сохранение результатов в NPZ-файл
    np.savez(args.output_file, data=results)
    print(f"Результаты сохранены в файл {args.output_file}")

if __name__ == "__main__":
    main()
