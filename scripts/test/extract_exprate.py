from jsonargparse import CLI


def main(err: int, path: str = "./Results_pred_symlg/Summary.txt"):
    struct_ln = None
    exprate_ln = None
    with open(path, "r") as f:
        for ln in f.readlines():
            if ln.startswith(" Structure"):
                struct_ln = ln
            if ln.startswith("Cum. Files"):
                exprate_ln = ln

    struct_rate = float(struct_ln.split()[1])
    correct_num = [int(x) for x in exprate_ln.split()[2 : 2 + err]]
    total_num = int(exprate_ln.split()[-1])

    print(f"Struct Rate: {struct_rate}")
    for i, n in enumerate(correct_num):
        print(f"Exprate {i} tolerated: {n / total_num * 100:.3f}")


if __name__ == "__main__":
    CLI(main)
