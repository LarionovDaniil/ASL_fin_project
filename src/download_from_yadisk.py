import zipfile
from pathlib import Path

import requests


def download_and_extract_nested_zip():
    zip_path = Path("asl_data_outer.zip")
    extract_path = Path("data_storage")

    if extract_path.exists():
        print("Данные уже распакованы. Пропускаем загрузку.")
        return

    print("Получаем ссылку с Яндекс.Диска...")
    url = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
    params = {"public_key": "https://disk.yandex.ru/d/co8ykC3973Jr3Q"}
    response = requests.get(url, params=params)
    response.raise_for_status()
    download_url = response.json()["href"]

    print("Скачиваем внешний архив...")
    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print("Распаковываем внешний архив...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    inner_zips = list(extract_path.rglob("*.zip"))
    if inner_zips:
        with zipfile.ZipFile(inner_zips[0], "r") as inner_zip:
            inner_zip.extractall(extract_path)
        inner_zips[0].unlink()
    zip_path.unlink()
    print("Всё готово! Данные лежат в папке 'data_storage/'.")


if __name__ == "__main__":
    download_and_extract_nested_zip()
