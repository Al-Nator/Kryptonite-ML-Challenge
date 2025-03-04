#!/usr/bin/env python3
"""
Скрипт скачивания моделей с Google Drive.

При наличии параметра --all скачиваются все модели (инференс и обучение).
Без параметра --all скачивается только модель для инференса.
"""

import os
import argparse
import gdown
from tqdm import tqdm


def parse_args():
    """Парсит аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Скрипт скачивания моделей с Google Drive."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Если указано, скачаются модели не только для инференса, но и для обучения."
    )
    return parser.parse_args()


def create_directories(directories):
    """Создает указанные директории, если их не существуют."""
    for d in directories:
        os.makedirs(d, exist_ok=True)


def download_files(files):
    """Загружает файлы по переданному словарю."""
    for path, file_id in tqdm(files.items(), desc="Загрузка файлов", unit="файл"):
        url = f"https://drive.google.com/uc?id={file_id}"
        tqdm.write(f"Загружаем {path}...")
        gdown.download(url, path, quiet=False)


def main():
    args = parse_args()

    files = {
        "models/checkpoints/ckpt_epoch1_batch20_acc0.9597_eer0.0282.ckpt": "13D-Zq6Eaydd3KfdOdIGCcHNcTeyXjtwo",  # https://drive.google.com/file/d/13D-Zq6Eaydd3KfdOdIGCcHNcTeyXjtwo
        "models/pretrained/adaface_ir101_webface12m.ckpt": "1XCcppcsdPFJX3fImvxbZ0ewQjAouqB4j",  # https://drive.google.com/file/d/1XCcppcsdPFJX3fImvxbZ0ewQjAouqB4j
        "models/pretrained/ckpt_epoch1_batch43000_acc0.9825_eer0.0165.ckpt": "1Bx4P51Xdl2Fl8RuT1XQ_-eJ7Yx3FATv3",  # https://drive.google.com/file/d/1Bx4P51Xdl2Fl8RuT1XQ_-eJ7Yx3FATv3
        "models/onnx/ckpt_epoch1_batch20_acc0.9597_eer0.0282.onnx": "10EQnwa94kZhcC8uZG56aSXfO_qjE8iCR",  # https://drive.google.com/file/d/10EQnwa94kZhcC8uZG56aSXfO_qjE8iCR
        "aligner/pretrained_model/model.pt": "17pOLtbze9E5SMkDfUNkXq5DU0TNPbNzr"  # https://drive.google.com/file/d/17pOLtbze9E5SMkDfUNkXq5DU0TNPbNzr
    }

    selected_files = (
        files if args.all else {p: fid for p, fid in files.items() if p.startswith("models/onnx/") or p.startswith("aligner/")}
    )

    required_dirs = {os.path.dirname(p) for p in selected_files}
    create_directories(required_dirs)

    download_files(selected_files)
    print("Загрузка завершена.")


if __name__ == "__main__":
    main()
