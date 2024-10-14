import os
import shutil
from argparse import ArgumentParser
from pathlib import Path


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("root")
    return vars(parser.parse_args())


def main(root):
    root = Path(root).expanduser()
    assert root.exists(), root.as_posix()
    print(root.as_posix())

    for item in os.listdir(root):
        shutil.move(root / item, root.parent / f"{root.name}_{item}")
    os.rmdir(root)
    print("fin")


if __name__ == "__main__":
    main(**parse_args())
