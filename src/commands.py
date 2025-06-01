import fire

from src.convert import main as convert_main
from src.infer import main as infer_main
from src.train import main as train_main


def train():
    train_main()


def infer():
    infer_main()


def convert():
    convert_main()


if __name__ == "__main__":
    fire.Fire(
        {
            "train": train,
            "infer": infer,
            "convert": convert,
        }
    )
