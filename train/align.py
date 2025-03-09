#!/usr/bin/env python3
"""
Модуль для выравнивания (align) датасета изображений и сохранения выровненных копий.
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import torchvision.transforms.functional as TF
from huggingface_hub import hf_hub_download
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from aligner.wrapper import CVLFaceAlignmentModel, ModelConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conf = None
aligner = None


def get_aligned_face(image_path):
    """
    Выравнивает лицо на изображении.

    Args:
        image_path (str): Путь к исходному изображению.

    Returns:
        PIL.Image: Выравненное изображение.
    """
    trans = Compose([
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img = Image.open(image_path).convert("RGB")
    input_tensor = trans(img).unsqueeze(0).to(device)
    output = aligner(input_tensor)
    aligned_tensor = output[0][0] 
    aligned_tensor = (aligned_tensor * 0.5 + 0.5).clamp(0, 1)
    aligned_img = TF.to_pil_image(aligned_tensor)
    return aligned_img


def save_aligned_dataset(original_dir, target_dir):
    """
    Выравнивает изображения из датасета и сохраняет их в указанной директории.
    Предполагается, что датасет организован в виде подпапок (например, пары изображений).

    Args:
        original_dir (str): Путь к исходной директории с изображениями.
        target_dir (str): Путь для сохранения выровненных изображений.
    """
    os.makedirs(target_dir, exist_ok=True)
    pair_ids = sorted(os.listdir(original_dir))
    for pair_id in tqdm(pair_ids, desc="Aligning dataset"):
        orig_pair_path = os.path.join(original_dir, pair_id)
        target_pair_path = os.path.join(target_dir, pair_id)
        os.makedirs(target_pair_path, exist_ok=True)
        # Перебираем все файлы в подпапке
        for fname in os.listdir(orig_pair_path):
            orig_path = os.path.join(orig_pair_path, fname)
            try:
                aligned_img = get_aligned_face(orig_path)
                aligned_img.save(os.path.join(target_pair_path, fname))
            except Exception as e:
                print(f"Error aligning {orig_path}: {e}")


def main():
    global aligner, conf
    
    parser = argparse.ArgumentParser(
        description="Выравнивание датасета изображений и сохранение выровненных копий."
    )
    parser.add_argument("--input_dir", type=str,
                        default="data/train/images",
                        help="Путь к исходной директории с изображениями")
    parser.add_argument("--output_dir", type=str,
                        default="data/train/images_aligned",
                        help="Путь для сохранения выровненных изображений")
    args = parser.parse_args()

    conf = ModelConfig()
    aligner = CVLFaceAlignmentModel(conf).to(device)
    aligner.eval()

    print(f"Выравнивание датасета из {args.input_dir} и сохранение в {args.output_dir}")
    save_aligned_dataset(args.input_dir, args.output_dir)
    print(f"Выровненный датасет сохранён в {args.output_dir}")


if __name__ == "__main__":
    main()
