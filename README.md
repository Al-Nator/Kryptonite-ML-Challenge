<p align="center">
  <img src="https://github.com/user-attachments/assets/81cdbc21-3b92-4116-b04c-4a746beb3e53" height="250">
</p>

<h1 align="center">
    Kryptonite ML Challenge<br>
    Команда MMG
</h1>

<br>

## 📝 Оглавление

<ol type="I">
  <li>
    <b>Шапка</b>
    <ul>
      <li><a href="#description">Описание задачи</a></li>
      <li><a href="#demo">Демо</a></li>
      <li><a href="#links">Ссылки</a></li>
      <li><a href="#data">Данные</a></li>
    </ul>
  </li>
  <li>
    <b>Проект</b>
    <ul>
      <li><a href="#tech-stack">Технологический стек</a></li>
      <li><a href="#project-structure">Структура проекта</a></li>
      <!-- <li><a href="#">Состав команды</a></li> -->
    </ul>
  </li>
  <li>
    <b>Установка</b>
    <ul>
      <li><a href="#cloning">Клонирование</a></li>
      <li><a href="#local">Запуск в локальной среде</a></li>
      <li><a href="#container">Запуск в контейнере</a></li>
    </ul>
  </li>
  <li>
    <b>Запуск скриптов</b>
    <ul>
      <li><a href="#inference">Инференс</a></li>
      <li><a href="#inference-page">Инференс на веб-сайте</a></li>
      <li><a href="#training">Обучение моделей</a></li>
      <li><a href="#onnx">Конвертация в onnx</a></li>
    </ul>
  </li>
  <li>
    <b>Планы на будущее</b>
    <ul>
      <li><a href="#scaling">Масштабирование системы</a></li>
    </ul>
  </li>
</ol>

<a id="description"></a>
## 🧐 Описание задачи
Поддельные изображения и видео, созданные с помощью технологии DeepFake, представляют угрозу для цифровой безопасности. Они могут быть настолько реалистичными, что их сложно отличить от настоящих.

Предстоит создать модель распознавания лиц, которая должна уметь:

* Сравнивать реальные фотографии одного и того же человека
* Различать снимки разных людей
* Распознавать фальшивые изображения, созданные с помощью DeepFake-технологий, без использования модулей защиты от спуфинга

<a id="demo"></a>
## 🎥 Демо
Демо решения запущено на ноутбуке с GeForce GTX 960M (2015г), протестировать его можно [тут](http://5.35.46.26:14500)
<div align="center">
  <img src="https://github.com/user-attachments/assets/1b655ab0-f6b3-480c-bc7f-4e3fc93a1b07" alt="demo" width="100%">
</div>

<a id="links"></a>
## 🔗 Ссылки
**Лендинг:** [kryptonite-ml.ru](https://kryptonite-ml.ru)  
**Репозиторий:** [git.codenrock.com/kryptonite-ml-challenge-1347](https://git.codenrock.com/kryptonite-ml-challenge-1347)  
**Тестирующая система:** [codenrock.com/contests/kryptonite-ml-challenge](https://codenrock.com/contests/kryptonite-ml-challenge)

<a id="data"></a>
## 📁 Данные
Перед началом работы необходимо загрузить данные и разместить их в папке `data`. 

- **Данные для обучения:** [Скачать по ссылке](https://storage.codenrock.com/companies/codenrock-13/contests/kryptonite-ml-challenge/train.zip)
- **Данные для теста:** [Скачать по ссылке](https://storage.codenrock.com/companies/codenrock-13/contests/kryptonite-ml-challenge/test_public.zip)
- **Данные для претрейна:** [Скачать по ссылке](https://drive.google.com/file/d/1muyIwX8c35Bl0OQTfmETGVjBevk3S6CK/view?usp=sharing)

<a id="tech-stack"></a>
## 🛠️ Технологический стек
- **Система:** Ubuntu 22.04
- **Инструменты контейнеризации:** Docker
- **Системы управления зависимостями:** conda, pip
- **Язык программирования:** Python 3.12.0
- **Глубокое обучение:** PyTorch
- **Работа с данными:** Transformers, TorchVision, OpenCV, NumPy, SciPy, timm
- **Веб-составляющая:** HTML, CSS, JavaScript, FastAPI, Uvicorn
- **Экспорт и инференс:** ONNX Runtime
- **CLI и утилиты:** argparse, tqdm

<a id="project-structure"></a>
## 📑 Структура проекта
```nushell
.
├── aligner <----------------------------- Модуль для выравнивания лиц
│   ├── aligners
│   ├── config.json <--------------------- Конфигурационный файл
│   ├── pretrained_model <---------------- Тут лежит предобученная модель для выравнивания
│   │   └── model.yaml
│   └── wrapper.py <---------------------- Обёртка для взаимодействия с моделью выравнивания
├── data <-------------------------------- Данные для обучения и тестирования
│   ├── test_public
│   └── train
├── models <------------------------------ Модели верификации
│   ├── checkpoints <--------------------- Контрольные точки обучения
│   ├── pretrained <---------------------- Предобученные модели
│   ├── onnx <---------------------------- Модели в формате ONNX
│   └── download_pretrained_models.py <--- Скрипт для скачивания моделей
├── scripts <----------------------------- Различные скрипты
│   ├── convert.py <---------------------- Скрипт конвертации моделей в ONNX
│   ├── inference.py <-------------------- Скрипт для инференса
│   └── utils <--------------------------- Вспомогательные модули
│       ├── net.py <---------------------- Определение архитектуры
│       └── utils.py <-------------------- Утилиты общего назначения
├── submissions <------------------------- Папка для сабмитов
├── web <--------------------------------- Веб-интерфейс
│   ├── main.py <------------------------- Основной скрипт веб-сервиса
│   └── static <-------------------------- Статические файлы
│       ├── images
│       ├── index.html
│       └── ...
├── Research-Report-MMG.pdf <------------- PDF Отчет о проделанном исследовании
├── docker-compose.yaml <----------------- Конфигурация для docker-compose
├── Dockerfile
├── environment.yml <--------------------- Список зависимостей Conda
├── LICENSE
├── README.md <--------------------------- Документация и инструкция по использованию
├── .dockerignore
└── .gitignore
```

<a id="cloning"></a>
## 📋 Клонирование
```nushell
git clone https://github.com/kekwak/Kryptonite-ML-Challenge.git
cd Kryptonite-ML-Challenge
```

<a id="local"></a>
## 💻 Запуск в локальной среде
В проекте используется анаконда.
*При установке зависимостей этим способом могут возникнуть проблемы.

```nushell
conda env create -f environment.yml && conda activate krypto
conda install conda-forge::onnxruntime
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
pip3 install numpy==2.2.3
```

<a id="container"></a>
## 🐳 Запуск в контейнере
Собрать контейнер самому:
```nushell
docker compose run --rm -it --service-ports app
```
*На данном шаге придется немного подождать...

Как только все создастся, активируйте среду для запуска скриптов:
```nushell
conda activate krypto
```
После этого можно запускать инференс в контейнере.

<a id="inference"></a>
## 🚀 Инференс

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

<a id="inference-page"></a>
## 🌐 Инференс на веб-сайте

Скачиваем все предобученные модели с гугл диска:
```nushell
python3 models/download_pretrained_models.py --all
```

Переходим в директорию `web` и запускаем наш сервер:
```nushell
cd web && python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```
Веб-страничка будет доступна по ссылке [0.0.0.0:8000](http://0.0.0.0:8000).

<a id="training"></a>
## 📚 Обучение моделей

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

<a id="scaling"></a>
## 📈 Масштабирование системы
- Использование более продвинутых и современных DeepFake и FaceSwap моделей
- Увеличение количества обучающих данных
- Верификация личности в реальном времени с помощью эталонных изображений
