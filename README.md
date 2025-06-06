# ASL_final_project

## ASL - American Sign Language

![alt text](image.png)

# Sign Language Gesture Classifier

Классификация жестов американского жестового языка (ASL) с использованием PyTorch Lightning, EfficientNetV2 и MLOps-инструментов.

## Описание проекта

Цель проекта — построить систему классификации жестов по изображению руки. В рамках проекта:

- Используется датасет с 20 статичными жестами (буквы и цифры)
- Реализована модель на EfficientNetV2 с PyTorch Lightning
- Поддерживаются этапы тренировки, инференса и конвертации в ONNX
- Используются Hydra, DVC, MLflow, pre-commit и Poetry

Train – 2000 фото jpg формата
Test – 1504 фото jpg формата
https://www.kaggle.com/competitions/sign-language-image-classification/overview
Данные с соревнования Kaggle по классификации знаков в языке жестов.


## Setup

1. Установи Poetry (если ещё нет):

```
curl -sSL https://install.python-poetry.org | python3 -
```

2. Клонируй репозиторий и установи зависимости:

```
git clone git@github.com:LarionovDaniil/ASL_fin_project.git
cd ASL_fin_project

poetry env use "путь до python.exe версии 3.9"
poetry install
```

3. Установи pre-commit и прогоните хуки:

```
poetry run pre-commit install
poetry run pre-commit run -a
```

## Train

1. Скачай данные:

```
poetry run python src/download_from_yadisk.py
```

2. MLflow

```
poetry run mlflow ui --port 8080
```

1. Тренировка модели

```
poetry run python src/train.py --config-name config
```

## Production preparation

1. Сохранение модели. Модель сохраняется в формате PyFunc: mlruns_model/sign_model/
```
poetry run python src/mlflow_wrapper.py
```
2. Запуск сервера
```
poetry run mlflow models serve -m mlruns_model/sign_model --no-conda -p 5000
```

## Infer
1. Запрос
Пример запроса:
Прописать абсолютный путь.
```
curl -X POST http://127.0.0.1:5000/invocations   -H "Content-Type: application/json"   -d '{
    "inputs": [
      {"image_path": "C:...ASL_fin_project\\\\data_storage\\\\images\\\\images\\\\test\\\\0ac0bb2730eb3123cdf48ba8fc5dcfe5.jpg"}
    ]
  }'
```

(Опционально) Работа модели на папке с фото, путь лежит в configs/infer/default.yaml
```
poetry run python src/infer.py
```


### Логирование

Для трекинга моделей используется MLflow.

адрес по умолчанию: http://127.0.0.1:8080

Графики и логи также сохраняются в папку plots

## Зависимости

Все зависимости управляются Poetry. Основные:

torch, pytorch-lightning, timm, hydra-core

mlflow, dvc, pre-commit, black, flake8, isort
