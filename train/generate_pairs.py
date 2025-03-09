#!/usr/bin/env python3
"""
Модуль для генерации пар изображений для обучения и валидации.
"""

import os
import json
import random
import itertools
import csv
import argparse
from tqdm import tqdm


def validate_paths(args):
    """Проверяет существование meta_path и папки images_root."""
    if not os.path.isfile(args.meta_path):
        raise FileNotFoundError(f"Файл {args.meta_path} не найден.")
    if not os.path.isdir(args.images_root):
        raise NotADirectoryError(f"Папка {args.images_root} не найдена.")


def load_meta(meta_path):
    """Загружает файл meta.json и возвращает словарь с метаданными."""
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta


def form_persons(meta):
    """
    Формирует словарь persons: для каждого person_id сохраняет списки файлов реальных и дипфейковых изображений.
    Формат ключа: person_id, значение: {"real": [файл], "fake": [файл]}.
    """
    persons = {}
    for key, value in meta.items():
        # key имеет формат "000000/0.jpg"
        person_id, filename = key.split("/")
        if person_id not in persons:
            persons[person_id] = {"real": [], "fake": []}
        if value == 0:
            persons[person_id]["real"].append(filename)
        else:
            persons[person_id]["fake"].append(filename)
    return persons


def split_persons(persons, train_split=0.95):
    """Разбивает список person_id на train и val наборы по заданной доле."""
    all_persons = sorted(persons.keys())
    random.shuffle(all_persons)
    split_idx = int(len(all_persons) * train_split)
    train_persons = all_persons[:split_idx]
    val_persons = all_persons[split_idx:]
    return train_persons, val_persons


def sample_positive_pairs(person_id, persons, images_root):
    """
    Для одного человека создаёт все позитивные пары (real-real, label=1) из реальных изображений.
    Возвращает список кортежей: (pathA, pathB, label, deepfake_a, deepfake_b).
    """
    real_imgs = [os.path.join(images_root, person_id, fname)
                 for fname in persons[person_id]["real"]]
    pos = []
    if len(real_imgs) >= 2:
        # Все комбинации по 2
        pos = list(itertools.combinations(real_imgs, 2))
        pos = [(p[0], p[1], "1", "0", "0") for p in pos]
    return pos


def sample_all_real_fake_pairs(person_id, persons, images_root):
    """
    Для одного человека, если имеются реальные и дипфейковые изображения,
    генерирует все возможные пары (real, fake) с label=0.
    Возвращает список кортежей: (pathA, pathB, label, deepfake_a, deepfake_b),
    где deepfake_a = 0 (real), deepfake_b = 1 (fake).
    """
    real_imgs = [os.path.join(images_root, person_id, fname)
                 for fname in persons[person_id]["real"]]
    fake_imgs = [os.path.join(images_root, person_id, fname)
                 for fname in persons[person_id]["fake"]]
    pairs = []
    if real_imgs and fake_imgs:
        for r in real_imgs:
            for f in fake_imgs:
                pairs.append((r, f, "0", "0", "1"))
    return pairs


def sample_real_another_real_pair(person_id, persons, images_root, other_persons):
    """
    Для одного человека выбирает одну негативную пару (real-another-real) – одну реальную фотографию из person_id и одну
    из случайного другого человека (у которого есть реальные изображения). Если нет подходящих, возвращает None.
    Возвращает кортеж: (pathA, pathB, label, deepfake_a, deepfake_b) с label=0 и deepfake_a=deepfake_b=0.
    """
    real_imgs_current = [os.path.join(images_root, person_id, fname)
                         for fname in persons[person_id]["real"]]
    if not real_imgs_current:
        return None
    candidates = [pid for pid in other_persons if pid != person_id and persons[pid]["real"]]
    if not candidates:
        return None
    other_pid = random.choice(candidates)
    real_imgs_other = [os.path.join(images_root, other_pid, fname)
                       for fname in persons[other_pid]["real"]]
    if not real_imgs_other:
        return None
    r1 = random.choice(real_imgs_current)
    r2 = random.choice(real_imgs_other)
    return (r1, r2, "0", "0", "0")


def create_pairs_for_person(person_id, persons, images_root, all_persons):
    """
    Для одного человека генерирует:
      - Все позитивные пары из реальных изображений (label=1, deepfake_a=0, deepfake_b=0).
      - Все негативные пары типа real-fake (label=0, deepfake_a=0, deepfake_b=1).
      - Для каждого позитивного примера генерирует по одной случайной негативной пары типа real-another-real (label=0, deepfake_a=0, deepfake_b=0).
    """
    pos_pairs = sample_positive_pairs(person_id, persons, images_root)
    neg_rf_pairs = sample_all_real_fake_pairs(person_id, persons, images_root)
    neg_rar_pairs = []
    if pos_pairs:
        for _ in range(len(pos_pairs)):
            pair_rar = sample_real_another_real_pair(person_id, persons, images_root, all_persons)
            if pair_rar is not None:
                neg_rar_pairs.append(pair_rar)
    return pos_pairs, neg_rf_pairs, neg_rar_pairs


def create_all_pairs(person_ids, persons, images_root):
    """
    Генерирует пары для набора людей.
    Возвращает три списка: позитивные пары, негативные пары (real-fake),
    негативные пары (real-another-real).
    """
    all_pos = []
    all_neg_rf = []
    all_neg_rar = []
    for pid in tqdm(person_ids, desc="Создание пар для набора"):
        pos, neg_rf, neg_rar = create_pairs_for_person(pid, persons, images_root, person_ids)
        all_pos.extend(pos)
        all_neg_rf.extend(neg_rf)
        all_neg_rar.extend(neg_rar)
    return all_pos, all_neg_rf, all_neg_rar


def save_pairs(csv_file, pairs):
    """Сохраняет пары в CSV файл с заголовком."""
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pathA", "pathB", "label", "deepfake_a", "deepfake_b"])
        for p in pairs:
            writer.writerow(p)


def count_categories(pairs):
    """
    Подсчитывает количество пар по категориям:
      - Real-Real: label == 1.
      - Real-Fake: label == 0 и обе фотографии принадлежат одному человеку.
      - Real-Another-Real: label == 0 и фотографии принадлежат разным людям.
    Возвращает словарь с подсчетами.
    """
    counts = {"Real-Real": 0, "Real-Fake": 0, "Real-Another-Real": 0}
    for a, b, label, _, _ in pairs:
        if int(label) == 1:
            counts["Real-Real"] += 1
        else:
            pid_a = os.path.basename(os.path.dirname(a))
            pid_b = os.path.basename(os.path.dirname(b))
            if pid_a == pid_b:
                counts["Real-Fake"] += 1
            else:
                counts["Real-Another-Real"] += 1
    return counts


def main(args):
    # Валидация путей
    validate_paths(args)

    # Загружаем meta.json
    meta = load_meta(args.meta_path)
    persons = form_persons(meta)

    # Разбиваем на train и val по указанной доле
    train_persons, val_persons = split_persons(persons, train_split=args.train_split)
    print(f"Всего людей: {len(persons)}. Train: {len(train_persons)}, Val: {len(val_persons)}")

    if set(train_persons).intersection(set(val_persons)):
        print("Ошибка: пересечение между train и val!")
    else:
        print("Train и Val не пересекаются.")

    # Генерируем пары для train и val
    train_pos, train_neg_rf, train_neg_rar = create_all_pairs(train_persons, persons, args.images_root)
    val_pos, val_neg_rf, val_neg_rar = create_all_pairs(val_persons, persons, args.images_root)

    # Балансировка: целевое количество негативных пар каждой категории = половине числа позитивных пар
    def balance_pairs(pos_pairs, neg_pairs):
        desired = len(pos_pairs) // 2
        if len(neg_pairs) > desired:
            return random.sample(neg_pairs, desired)
        return neg_pairs

    train_neg_rf_bal = balance_pairs(train_pos, train_neg_rf)
    train_neg_rar_bal = balance_pairs(train_pos, train_neg_rar)
    val_neg_rf_bal = balance_pairs(val_pos, val_neg_rf)
    val_neg_rar_bal = balance_pairs(val_pos, val_neg_rar)

    # Объединяем пары
    train_pairs = train_pos + train_neg_rf_bal + train_neg_rar_bal
    val_pairs = val_pos + val_neg_rf_bal + val_neg_rar_bal

    # Сохраняем в CSV
    save_pairs(args.output_train, train_pairs)
    save_pairs(args.output_val, val_pairs)

    # Выводим баланс по категориям
    def print_balance(name, pairs):
        total = len(pairs)
        counts = count_categories(pairs)
        print(f"\n{name} распределение:")
        print(f"Общее число пар: {total}")
        for cat, cnt in counts.items():
            print(f"  {cat}: {cnt} ({(cnt/total)*100:.2f}%)")

    print_balance("Train", train_pairs)
    print_balance("Val", val_pairs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Генерация пар изображений для train и val наборов на основе meta.json и изображений."
    )
    parser.add_argument(
        "--meta_path",
        type=str,
        default="data/train/meta.json",
        help="Путь к файлу meta.json"
    )
    parser.add_argument(
        "--images_root",
        type=str,
        default="data/train/images",
        help="Путь к корневой папке изображений"
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.95,
        help="Доля людей, отведённая под train (остальные – val)"
    )
    parser.add_argument(
        "--output_train",
        type=str,
        default="data/train/train_pairs.csv",
        help="Путь для сохранения CSV файла с train парами"
    )
    parser.add_argument(
        "--output_val",
        type=str,
        default="data/train/val_pairs.csv",
        help="Путь для сохранения CSV файла с val парами"
    )
    args = parser.parse_args()
    main(args)
