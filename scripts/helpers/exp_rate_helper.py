from os import path
from typing import List

from jsonargparse import CLI


BASE_DIR = ""
# extract ExpRate-0 from multiple versions and years and prints them below each other to create a table
def main(base: str, versions: List[int], years=None):
    if years is None:
        years = ["2014", "2016", "2019"]

    for version in versions:
        for i, year in enumerate(years):
            with open(path.join(BASE_DIR, base, f"version_{version}", f"{year}.txt")) as f:
                lines = f.readlines()
                exp_zero = float(lines[1].split()[-1])
                print(f"${exp_zero:.1f}$", flush=True, end='')
            if i is not len(years) - 1:
                print(" & ", flush=True, end='')
        print()


if __name__ == "__main__":
    CLI(main)