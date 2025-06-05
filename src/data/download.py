import subprocess


def download_data():
    """
    Загружает данные из DVC-хранилища. Работает через CLI.
    """
    print("Загружаем данные через DVC...")
    try:
        subprocess.run(["dvc", "pull", "data.dvc"], check=True)
        print("Данные успешно загружены.")
    except subprocess.CalledProcessError:
        print("Ошибка при выполнении dvc pull.")
