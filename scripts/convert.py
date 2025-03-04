#!/usr/bin/env python3
"""
Скрипт для экспорта модели в формат ONNX.

Данный скрипт загружает PyTorch-модель по указанному чекпоинту,
оборачивает её в AdaFaceONNXWrapper и экспортирует в формат ONNX.
"""

import torch
import argparse
from utils.utils import load_pretrained_model, load_checkpoint, AdaFaceONNXWrapper


def convert_to_onnx(checkpoint_path, output_model, architecture="ir_101", input_size=(112, 112)):
    """
    Экспортирует модель в формат ONNX.

    Аргументы:
        checkpoint_path (str): Путь к PyTorch-чекпоинту.
        output_model (str): Имя выходного ONNX-файла.
        architecture (str, optional): Архитектура модели. По умолчанию "ir_101".
        input_size (tuple, optional): Размер входного изображения. По умолчанию (112, 112).
    """
    print("Загрузка PyTorch-модели AdaFace...")
    model = load_pretrained_model(architecture, load_weights=False)
    model = load_checkpoint(checkpoint_path, model)
    model.eval()
    model.to("cpu")

    wrapper = AdaFaceONNXWrapper(model)
    wrapper.eval()

    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
    print("Экспорт модели в формат ONNX...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_model,
        input_names=["input"],
        output_names=["features"],
        dynamic_axes={"input": {0: "batch_size"}, "features": {0: "batch_size"}},
        opset_version=11
    )
    print(f"Модель успешно экспортирована в {output_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Экспорт модели в формат ONNX.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Путь к PyTorch-чекпоинту."
    )
    parser.add_argument(
        "--output_model",
        type=str,
        default="models/onnx/verification_model.onnx",
        help="Имя выходного ONNX-файла."
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="ir_101",
        help="Архитектура модели."
    )
    args = parser.parse_args()
    convert_to_onnx(args.checkpoint, args.output_model, args.architecture)
