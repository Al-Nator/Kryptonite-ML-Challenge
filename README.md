<p align="center">
    <img src="https://github.com/user-attachments/assets/81cdbc21-3b92-4116-b04c-4a746beb3e53" height="250">
</p>

<h1 align="center">
    Kryptonite ML Challenge 
    <br>
    Команда MMG
</h1>

<br>

## 📝 Оглавление

1. [Шапка](#demo)
    - [Демо](#demo)
    - [Описание задачи](#description)
    - [Данные](#data)
    - [Ссылки](#links)
    - [Технологический стек](#tech-stack)
    <!-- - [Состав команды]() -->
2. [Установка и запуск](#cloning)
    - [Клонирование](#cloning)
    - [Запуск в контейнере](#container)
    - [Запуск в локальной среде](#local)
    - [Инференс](#inference)
    - [Инференс на веб-сайте](#inference-page)
    - [Обучение](#training)
    - [Конвертация в onnx](#onnx)
3. [Структура проекта](#project-structure)
    - [Масштабирование](#scaling)

<a id="demo"></a>
## 🎥 Демо
<видео-gif тут>

<a id="description"></a>
## 🧐 Описание задачи
Поддельные изображения и видео, созданные с помощью технологии DeepFake, представляют угрозу для цифровой безопасности. Они могут быть настолько реалистичными, что их сложно отличить от настоящих.

Предстоит создать модель распознавания лиц, которая должна уметь:

* сравнивать реальные фотографии одного и того же человека
* различать снимки разных людей
* распознавать фальшивые изображения, созданные с помощью DeepFake-технологий, без использования модулей защиты от спуфинга

<a id="data"></a>
### 📁 Данные
Перед началом работы необходимо загрузить данные и разместить их в папке `data`. 

- **Данные для обучения**: [Скачать по ссылке](https://storage.codenrock.com/companies/codenrock-13/contests/kryptonite-ml-challenge/train.zip)
- **Данные для теста**: [Скачать по ссылке](https://storage.codenrock.com/companies/codenrock-13/contests/kryptonite-ml-challenge/test_public.zip)

<a id="links"></a>
## 🔗 Ссылки
Лендинг: [kryptonite-ml.ru](https://kryptonite-ml.ru)  
Репозиторий: [git.codenrock.com/kryptonite-ml-challenge-1347](https://git.codenrock.com/kryptonite-ml-challenge-1347)  
Тестирующая система: [codenrock.com/contests/kryptonite-ml-challenge](https://codenrock.com/contests/kryptonite-ml-challenge/)

<a id="tech-stack"></a>
## 🛠 Технологический стек
!!! ТРЕБУЕТСЯ ДОПОЛНЕНИЕ !!!
- **Система:** Ubuntu 22.04
- **Язык:** Python 3.12+
- **Фреймворк глубокого обучения:** PyTorch
- **Экспорт модели:** ONNX
- **Инференс:** ONNX Runtime
- **Выравнивание лиц:** Пользовательский модуль
- **Обработка данных:** PIL, NumPy, torchvision
- **CLI и утилиты:** argparse, tqdm

<a id="cloning"></a>
## 📋 Клонирование
```nushell
git clone https://github.com/kekwak/Kryptonite-ML-Challenge.git
cd Kryptonite-ML-Challenge
```

<a id="container"></a>
## 🐳 Запуск в контейнере
Собрать контейнер самому:
```nushell
...
```
или скачать готовый образ:
```nushell
...
```
а также запуск через альтернативный докерфайл:
```nushell
...
```

<a id="local"></a>
## 🖥 Запуск в локальной среде
В проекте используется анаконда.  
* При установке зависимостей этим способом могут возникнуть проблемы.

```nushell
conda env create -f environment.yml && conda activate krypto
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
pip3 install numpy==2.2.3
```

<a id="inference"></a>
## 🤖 Инференс

Для начала нужно скачать обученную модель с гугл диска. Это можно сделать вручную или использовать скрипт для загрузки моделей без аргумента `--all`:
```nushell
python3 models/download_pretrained_models.py
```

Публичный или приватный датасет нужно разместить в папке **data**, вот таким образом:
```nushell
data
└── test_public
   └── images
       ├── 00000000
       │   ├── 0.jpg
       │   └── 1.jpg
       ├── 00000001
       │   ├── 0.jpg
       │   └── 1.jpg
       ├── ...
       ...
```

Как только предыдущие шаги будут выполнены, можно приступать к инференсу. Он запускается следующей командой:
```nushell
python3 scripts/inference.py --model models/onnx/ckpt_epoch1_batch20_acc0.9597_eer0.0282.onnx
```
В аргументах достаточно указать только модель, но также есть возможность указать другую папку с изображениями с помощью аргумента `--input_dir data/path/to/images`.

*В инференсе используется [aligner](https://huggingface.co/minchul/cvlface_DFA_resnet50) для обрезания и выравнивания изображений.

<a id="inference"></a>
## 🌐 Инференс на веб-сайте

Скачиваем все предобученные модели с гугл диска:
```nushell
python3 models/download_pretrained_models.py --all
```

Переходим в директорию `web` и запускаем наш сервер:
```nushell
cd web && python3 -m uvicorn main:app --host 0.0.0.0
```
Веб-страничка будет длступна по ссылке [0.0.0.0:8000](http://0.0.0.0:8000)

<a id="training"></a>
## 📚 Обучение

Перед началом обучения нужно скачать все предобученные модели с гугл диска. Это можно сделать вручную или использовать скрипт для загрузки моделей с аргументом `--all`:
```nushell
python3 models/download_pretrained_models.py --all
```
Уже скачанные модели перезапишутся.

...

<a id="onnx"></a>
## 🔄 Конвертация в onnx

Для конвертации обученной модели в onnx формат достаточно использовать команду:
```nushell
python3 scripts/convert.py --checkpoint path/to/model.ckpt --output_model path/to/onnx_model.onnx
```

<a id="project-structure"></a>
## 🗂 Структура проекта
```nushell
.
├── app
│   ├── __init__.py
...
```

<a id="scaling"></a>
## 📈 Планы масштабирования системы
- Использование более продвинутых и современных DeepFake и FaceSwap моделей
- Увеличение количества обучающих данных
- Верификация личности в реальном времени с помощью эталонных изображений
