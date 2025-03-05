#!/usr/bin/env python3
"""
Унифицированный скрипт, объединяющий выравнивание и инференс ONNX в одном пайплайне.
Результаты сохраняются в CSV.
"""

import os
import sys
import time
import csv
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Добавляем путь для доступа к модулю aligner
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
import torch
torch.backends.cudnn.benchmark = True

import onnxruntime as ort

from torchvision import transforms
from aligner.wrapper import CVLFaceAlignmentModel, ModelConfig

import warnings
warnings.filterwarnings("ignore")

# Глобальные переменные
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALIGNER = None


# Функция для загрузки и преобразования изображения через OpenCV
def load_and_transform_cv2(path, target_size=(112,112)):
    """
    Считывает изображение через cv2, преобразует BGR->RGB, меняет размер до target_size,
    переводит в float и масштабирует в диапазон [0,1], затем нормализует: (x-0.5)/0.5 (=> [-1,1]).
    Возвращает тензор размера (3, H, W).
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return torch.zeros(3, target_size[0], target_size[1], dtype=torch.float32)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img_rgb.shape[:2] != target_size:
        img_rgb = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(img_rgb).permute(2,0,1).float() / 255.0
    return (tensor - 0.5) / 0.5


def load_and_align_chunk(chunk_pairs, align_batch_size, num_load_workers):
    """
    Для списка пар (pair_id, path0, path1):
      - Собирает пары
      - Загружает изображения параллельно (число воркеров = num_load_workers).
      - Разбивает полученные тензоры на мини-батчи размера align_batch_size,
        прогоняет их через модель ALIGNER (на GPU) с torch.inference_mode(),
        затем преобразует результат для инференса.
      - Собирает результат в numpy массив.
    """
    # Формируем список
    ordered = []
    for pair in chunk_pairs:
        pid, p0, p1 = pair
        ordered.append((pid, p0))
        ordered.append((pid, p1))
    # Загружаем изображения параллельно
    tensors = []
    with ThreadPoolExecutor(max_workers=num_load_workers) as executor:
        futures = [executor.submit(load_and_transform_cv2, path, target_size=(112,112))
                   for (_, path) in ordered]
        for f in futures:
            tensors.append(f.result())
    # Разбиваем на мини-батчи для выравнивания
    aligned_batches = []
    for i in range(0, len(tensors), align_batch_size):
        batch_tensors = tensors[i:i+align_batch_size]
        batch = torch.stack(batch_tensors).to(DEVICE)
        with torch.inference_mode():
            aligned_batch, _, _, _, _, _ = ALIGNER(batch)
        # Приводим к диапазону [0,1]
        aligned_batch = (aligned_batch * 0.5 + 0.5).clamp(0,1)
        # Преобразуем для инференса и меняем порядок каналов с RGB на BGR
        inf_batch = (aligned_batch - 0.5) / 0.5
        inf_batch = inf_batch[:, [2,1,0], :, :]
        aligned_batches.append(inf_batch.cpu().numpy())
    if aligned_batches:
        big_arr = np.concatenate(aligned_batches, axis=0)
    else:
        big_arr = np.empty((0,3,112,112), dtype=np.float32)
    return ordered, big_arr


def create_ort_session(model_path):
    """
    Создаёт сессию ONNX Runtime для инференса модели.
    """
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = [
        (
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
            },
        ),
        "CPUExecutionProvider",
    ]
    session = ort.InferenceSession(model_path, sess_options=options, providers=providers)
    input_name = session.get_inputs()[0].name
    return session, input_name


def unified_inference(model_path, input_dir, batch_size, chunk_size,
                      align_batch_size, num_load_workers, num_infer_workers, chunk_workers):
    """
    Унифицированный режим:
      - Читает input_dir, где каждая подпапка - pair_id с файлами 0.jpg и 1.jpg.
      - Обрабатывает пары чанками:
           • Для каждого чанка загружает и выравнивает изображения,
           • Делит выровненные данные на мини-батчи для ONNX-инференса,
           • Вычисляет эмбеддинги, нормализует их и считает косинусное сходство для каждой пары.
      - Результаты сохраняются в CSV.
    """
    session, input_name = create_ort_session(model_path)
    # Собираем пары
    pair_ids = sorted(os.listdir(input_dir))
    pairs = []
    for pid in pair_ids:
        pair_folder = os.path.join(input_dir, pid)
        if not os.path.isdir(pair_folder):
            continue
        p0 = os.path.join(pair_folder, "0.jpg")
        p1 = os.path.join(pair_folder, "1.jpg")
        if os.path.exists(p0) and os.path.exists(p1):
            pairs.append((pid, p0, p1))
    print(f"Найдено {len(pairs)} пар изображений.")

    submission = []
    num_chunks = (len(pairs) + chunk_size - 1) // chunk_size

    # Используем пул для предвыборки чанков
    with ThreadPoolExecutor(max_workers=chunk_workers) as prefetch_exec:
        next_chunk_future = prefetch_exec.submit(load_and_align_chunk, pairs[:chunk_size], align_batch_size, num_load_workers)
        for chunk_index in tqdm(range(num_chunks), desc="Processing chunks"):
            start_idx = chunk_index * chunk_size
            end_idx = min((chunk_index + 1) * chunk_size, len(pairs))
            ordered, imgs_array = next_chunk_future.result()
            if end_idx < len(pairs):
                next_chunk_future = prefetch_exec.submit(load_and_align_chunk,
                                                         pairs[end_idx:end_idx+chunk_size],
                                                         align_batch_size,
                                                         num_load_workers)
            else:
                next_chunk_future = None

            num_samples = imgs_array.shape[0]
            # ONNX инференс параллельно
            outputs_list = []
            with ThreadPoolExecutor(max_workers=num_infer_workers) as inf_ex:
                futs = []
                for i in range(0, num_samples, batch_size):
                    batch_data = imgs_array[i:i+batch_size]
                    futs.append(inf_ex.submit(session.run, None, {input_name: batch_data}))
                for f in futs:
                    outputs_list.append(f.result()[0])
            embeddings_chunk = np.concatenate(outputs_list, axis=0)
            norms = np.linalg.norm(embeddings_chunk, axis=1, keepdims=True) + 1e-6
            embeddings_norm = embeddings_chunk / norms
            emb_pairs = embeddings_norm.reshape(-1, 2, embeddings_norm.shape[-1])
            sims = np.sum(emb_pairs[:, 0, :] * emb_pairs[:, 1, :], axis=1)
            num_pairs = len(ordered) // 2
            for j in range(num_pairs):
                pid = ordered[2*j][0]
                submission.append((pid, sims[j]))

    timestamp = int(time.time())
    out_csv = f"submissions/submission_{timestamp}.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "similarity"])
        for pid, sim in submission:
            writer.writerow([pid, sim])
    print(f"Результаты сохранены в {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Унифицированный инференс (выравнивание + ONNX)")
    parser.add_argument("--model", type=str, required=True, help="Путь к ONNX модели.")
    parser.add_argument("--input_dir", type=str, default="data/test_public/images",
                        help="Путь к папке с исходными изображениями (подпапки с 0.jpg и 1.jpg).")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Размер подбатча для ONNX инференса.")
    parser.add_argument("--chunk_size", type=int, default=32,
                        help="Количество пар для обработки за один чанк.")
    parser.add_argument("--align_batch_size", type=int, default=128,
                        help="Размер батча для выравнивания (минимальные батчи).")
    parser.add_argument("--num_load_workers", type=int, default=8,
                        help="Количество воркеров для параллельной загрузки изображений.")
    parser.add_argument("--num_infer_workers", type=int, default=8,
                        help="Количество воркеров для параллельного инференса ONNX.")
    parser.add_argument("--chunk_workers", type=int, default=16,
                        help="Количество воркеров для параллельной предвыборки чанков.")
    args = parser.parse_args()

    # Инициализируем глобальный ALIGNER
    conf = ModelConfig()
    ALIGNER = CVLFaceAlignmentModel(conf).to(DEVICE)
    ALIGNER.eval()

    unified_inference(args.model, args.input_dir, args.batch_size, args.chunk_size,
                      args.align_batch_size, args.num_load_workers, args.num_infer_workers, args.chunk_workers)
