import numpy as np
from jsonargparse import CLI
import pathlib

def main(path: str):
    root = pathlib.Path(path)
    exp = []
    for years in [2014, 2016, 2019]:
        p = root.joinpath(f'{years}.txt')
        with open(p, "r") as f:
            for ln in f.readlines():
                if ln.startswith("Exprate 0 tolerated: "):
                    exp.append(float(ln[len("Exprate 0 tolerated: "):]))

    print(' & '.join([f'\onslide<5->{{${x:.2f}$}}' for x in np.array(exp)]))


if __name__ == "__main__":
    CLI(main)
