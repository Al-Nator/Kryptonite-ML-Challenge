import torch
from . import net


def load_pretrained_model(architecture="ir_101", load_weights=False):
    """
    Загружает предобученную модель.

    Аргументы:
        architecture (str): Архитектура модели. По умолчанию "ir_101".
        load_weights (bool): Если True, загружает веса модели.

    Возвращает:
        model (torch.nn.Module): Загруженная модель.
    """
    adaface_models = {
        "ir_101": "models/pretrained/adaface_ir101_webface12m.ckpt",
    }

    if architecture not in adaface_models:
        raise ValueError(f"Архитектура {architecture} не поддерживается.")

    model = net.build_model(architecture)
    if load_weights:
        checkpoint = torch.load(adaface_models[architecture], weights_only=False)
        statedict = checkpoint["state_dict"]
        # Удаляем префикс "model." из ключей
        model_statedict = {key[6:]: val for key, val in statedict.items() if key.startswith("model.")}
        model.load_state_dict(model_statedict)

    model.eval()
    return model


def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """
    Загружает чекпоинт и обновляет состояние модели, оптимизатора и scheduler.

    Аргументы:
        filepath (str): Путь к файлу чекпоинта.
        model (torch.nn.Module): Модель, которую нужно загрузить.
        optimizer (torch.optim.Optimizer, optional): Оптимизатор.
        scheduler (torch.optim.lr_scheduler, optional): Планировщик скорости обучения.

    Возвращает:
        model (torch.nn.Module): Модель с загруженными весами.
    """
    checkpoint = torch.load(filepath, weights_only=False, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch", -1)
    global_batch = checkpoint.get("global_batch", -1)
    print(f"Чекпоинт загружен: epoch {epoch}, global batch {global_batch}")

    return model


class AdaFaceONNXWrapper(torch.nn.Module):
    """
    Обертка для модели.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        feature, _ = self.model(x)
        return feature
