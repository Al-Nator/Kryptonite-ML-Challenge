#!/usr/bin/env python3
"""
Модуль для разделения датасета по кластерной логике, создания коллажей и генерации CSV-файла.
"""

import os
import json
import numpy as np
import igraph as ig
import shutil
import csv
import argparse
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def draw_text_with_outline(draw, position, text, font,
                           text_color=(255, 255, 255),
                           outline_color=(0, 0, 0), thickness=1):
    """
    Рисует текст с обводкой (чёрная обводка, белый текст).

    :param draw: объект ImageDraw.Draw
    :param position: кортеж (x, y) – координаты левого верхнего угла текста
    :param text: сам текст
    :param font: шрифт для отрисовки
    :param text_color: цвет текста
    :param outline_color: цвет обводки
    :param thickness: толщина обводки
    """
    x, y = position
    for dx in range(-thickness, thickness + 1):
        for dy in range(-thickness, thickness + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
    draw.text(position, text, font=font, fill=text_color)


def create_collage_with_labels(left_paths, right_paths, collage_save_path,
                               thumb_size=(128, 128)):
    """
    Создаёт коллаж, где левая колонка содержит изображения из left_paths,
    а правая – из right_paths. На каждом изображении в левом нижнем углу наносится
    подпись (имя файла) белым текстом с чёрной обводкой.

    :param left_paths: список путей к изображениям (левая колонка)
    :param right_paths: список путей к изображениям (правая колонка)
    :param collage_save_path: путь для сохранения итогового коллажа
    :param thumb_size: размер миниатюры (ширина, высота)
    """
    font = ImageFont.load_default()

    def load_and_label(path):
        img = Image.open(path).convert("RGB").resize(thumb_size)
        draw = ImageDraw.Draw(img)
        filename = os.path.basename(path)
        text_position = (5, thumb_size[1] - 20)
        draw_text_with_outline(draw, text_position, filename, font,
                               text_color=(255, 255, 255),
                               outline_color=(0, 0, 0),
                               thickness=1)
        return img

    left_images = [load_and_label(p) for p in left_paths]
    right_images = [load_and_label(p) for p in right_paths]

    max_rows = max(len(left_images), len(right_images))
    blank = Image.new("RGB", thumb_size, (255, 255, 255))

    while len(left_images) < max_rows:
        left_images.append(blank)
    while len(right_images) < max_rows:
        right_images.append(blank)

    col_width, col_height = thumb_size[0], max_rows * thumb_size[1]
    left_column = Image.new("RGB", (col_width, col_height), (255, 255, 255))
    right_column = Image.new("RGB", (col_width, col_height), (255, 255, 255))

    for i, img in enumerate(left_images):
        left_column.paste(img, (0, i * thumb_size[1]))
    for i, img in enumerate(right_images):
        right_column.paste(img, (0, i * thumb_size[1]))

    collage = Image.new("RGB", (col_width * 2, col_height), (255, 255, 255))
    collage.paste(left_column, (0, 0))
    collage.paste(right_column, (col_width, 0))
    collage.save(collage_save_path)


def main(orig_dir,
         npz_file,
         cleaned_dir,
         collage_dir,
         csv_file,
         meta_path,
         new_meta_path,
         threshold):
    """
    Основная функция, сохраняющая исходную логику разделения датасета по
    кластерной логике, создания коллажей и генерации CSV-файла.

    :param orig_dir: исходная директория с изображениями
    :param npz_file: путь к NPZ-файлу с результатами cosine similarity
    :param cleaned_dir: путь к новой директории (разделённой по кластерной логике)
    :param collage_dir: путь к директории для сохранения коллажей
    :param csv_file: путь к CSV-файлу с информацией о переносах
    :param meta_path: путь к исходному meta.json
    :param new_meta_path: путь к выходному meta.json (очищенный/обновлённый)
    :param threshold: пороговое значение для cosine similarity
    """

    # Создаём необходимые директории
    os.makedirs(cleaned_dir, exist_ok=True)
    os.makedirs(collage_dir, exist_ok=True)

    # Загружаем meta.json
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Загружаем NPZ-файл с результатами (для всех фото)
    npz_data = np.load(npz_file, allow_pickle=True)
    npz_results = npz_data["data"].item()

    new_meta = {}
    # Для формирования CSV: {person: {filename: flag}}, flag = 1, если файл перемещён в новую папку, иначе 0
    csv_data = {}

    # Список папок (person ID)
    person_folders = [
        d for d in os.listdir(orig_dir)
        if os.path.isdir(os.path.join(orig_dir, d))
    ]

    for person in tqdm(person_folders, desc="Processing persons"):
        person_orig_path = os.path.join(orig_dir, person)
        # Создаем основную папку для данного лица
        person_cleaned_path = os.path.join(cleaned_dir, person)
        os.makedirs(person_cleaned_path, exist_ok=True)
        # Папка для перенесённых файлов (новая) создается только при необходимости
        person_new_path = os.path.join(cleaned_dir, person + "_new")

        all_files = [
            f for f in os.listdir(person_orig_path)
            if f.lower().endswith('.jpg')
        ]
        csv_data[person] = {}

        # Списки для разделения
        real_kept = []    # реальные фото, попавшие в основную компоненту
        real_deleted = [] # реальные фото, не связанные с основной компонентой (будут перемещены)
        fake_main = []    # дипфейки, остающиеся в основной папке
        fake_new = []     # дипфейки, перемещаемые в новую папку

        # Если имеются NPZ-результаты для данного лица
        if person in npz_results:
            cosine_matrix = npz_results[person]['cosine_matrix']
            names = npz_results[person]['file_names']
            name_to_index = {name: idx for idx, name in enumerate(names)}

            # Строим граф для реальных фото (label == 0)
            real_indices = [
                i for i, name in enumerate(names)
                if meta.get(f"{person}/{name}", 1) == 0
            ]
            edges = []
            weights = []
            for i in range(len(real_indices)):
                for j in range(i + 1, len(real_indices)):
                    idx_i = real_indices[i]
                    idx_j = real_indices[j]
                    if cosine_matrix[idx_i, idx_j] >= threshold:
                        edges.append((i, j))
                        weights.append(cosine_matrix[idx_i, idx_j])

            g = ig.Graph()
            g.add_vertices(len(real_indices))
            if edges:
                g.add_edges(edges)
                g.es['weight'] = weights
            real_names = [names[i] for i in real_indices]
            g.vs["name"] = real_names

            # Удаляем изолированные вершины
            isolated = [v.index for v in g.vs if g.degree(v) == 0]
            if isolated:
                g.delete_vertices(isolated)

            # Если граф состоит из нескольких компонент, оставляем только наибольшую
            components = g.connected_components()
            if len(components) > 1:
                largest_component = max(components, key=len)
                vertices_to_remove = [
                    v.index for v in g.vs if v.index not in largest_component
                ]
                g.delete_vertices(vertices_to_remove)
            cleaned_names = set(g.vs["name"])
        else:
            cleaned_names = None
            name_to_index = {}

        # Обрабатываем каждый файл
        for filename in all_files:
            key = f"{person}/{filename}"
            label = meta.get(key, 1)  # Если нет – считаем дипфейком
            src = os.path.join(person_orig_path, filename)

            # Если реальное фото
            if label == 0:
                if cleaned_names is not None:
                    if filename in cleaned_names:
                        real_kept.append(filename)
                        dest = os.path.join(person_cleaned_path, filename)
                        csv_data[person][filename] = 0
                    else:
                        real_deleted.append(filename)
                        os.makedirs(person_new_path, exist_ok=True)
                        dest = os.path.join(person_new_path, filename)
                        csv_data[person][filename] = 1
                else:
                    real_kept.append(filename)
                    dest = os.path.join(person_cleaned_path, filename)
                    csv_data[person][filename] = 0

            else:
                # Для дипфейков – если NPZ-результаты есть и имя найдено,
                # сравниваем с реальными из основной компоненты
                if cleaned_names is not None and filename in name_to_index:
                    fake_idx = name_to_index[filename]
                    max_sim = 0.0
                    for real_file in cleaned_names:
                        if real_file in name_to_index:
                            sim = cosine_matrix[name_to_index[real_file], fake_idx]
                            if sim > max_sim:
                                max_sim = sim
                    if max_sim >= threshold:
                        fake_main.append(filename)
                        dest = os.path.join(person_cleaned_path, filename)
                        csv_data[person][filename] = 0
                    else:
                        fake_new.append(filename)
                        os.makedirs(person_new_path, exist_ok=True)
                        dest = os.path.join(person_new_path, filename)
                        csv_data[person][filename] = 1
                else:
                    fake_main.append(filename)
                    dest = os.path.join(person_cleaned_path, filename)
                    csv_data[person][filename] = 0

            shutil.copy(src, dest)
            new_meta[f"{person}/{filename}"] = label

        # Создаем коллаж, если новая папка создана и содержит файлы (т.е. если были перемещения)
        if os.path.exists(person_new_path) and os.listdir(person_new_path):
            # Коллаж содержит все фото: слева – из основной папки, справа – из новой
            left_paths = [
                os.path.join(person_cleaned_path, f)
                for f in os.listdir(person_cleaned_path)
                if f.lower().endswith('.jpg')
            ]
            right_paths = [
                os.path.join(person_new_path, f)
                for f in os.listdir(person_new_path)
                if f.lower().endswith('.jpg')
            ]
            collage_filename = f"{person}.jpg"  # Имя коллажа = название папки
            collage_save_path = os.path.join(collage_dir, collage_filename)
            create_collage_with_labels(left_paths, right_paths,
                                       collage_save_path,
                                       thumb_size=(128, 128))

    # Сохраняем новый meta.json
    with open(new_meta_path, "w", encoding="utf-8") as f:
        json.dump(new_meta, f, indent=2)

    # Генерация CSV-файла
    csv_columns = [f"{i}.jpg" for i in range(11)]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["person"] + csv_columns
        writer.writerow(header)
        for person, files_dict in csv_data.items():
            row = [person]
            # Для каждого столбца, если файл присутствует в словаре, ставим 1 или 0,
            # иначе оставляем пустым
            for col in csv_columns:
                row.append(files_dict.get(col, ""))
            writer.writerow(row)

    print("Новый датасет разделён по папкам, коллажи созданы, "
          "а CSV с информацией о переносах сгенерирован.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Разделение датасета по кластерной логике, создание коллажей "
            "и генерация CSV-файла с исходной логикой."
        )
    )
    parser.add_argument(
        "--orig-dir", type=str, default="data/train/images_aligned",
        help="Путь к исходному датасету"
    )
    parser.add_argument(
        "--npz-file", type=str, default="data/cosine_similarity_results.npz",
        help="Путь к NPZ-файлу с результатами cosine similarity"
    )
    parser.add_argument(
        "--cleaned-dir", type=str,
        default="data/train/images_aligned_filtered",
        help="Новый датасет, разделённый по кластерной логике"
    )
    parser.add_argument(
        "--collage-dir", type=str, default="data/collages",
        help="Папка для сохранения коллажей"
    )
    parser.add_argument(
        "--csv-file", type=str, default="data/output.csv",
        help="CSV с информацией о переносе файлов"
    )
    parser.add_argument(
        "--meta-path", type=str, default="data/train/meta.json",
        help="Путь к исходному meta.json"
    )
    parser.add_argument(
        "--new-meta-path", type=str,
        default="data/train/meta_filtered.json",
        help="Путь к новому meta.json"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.15,
        help="Порог для cosine similarity"
    )

    args = parser.parse_args()

    main(
        orig_dir=args.orig_dir,
        npz_file=args.npz_file,
        cleaned_dir=args.cleaned_dir,
        collage_dir=args.collage_dir,
        csv_file=args.csv_file,
        meta_path=args.meta_path,
        new_meta_path=args.new_meta_path,
        threshold=args.threshold
    )
