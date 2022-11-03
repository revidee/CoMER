import time
from typing import List, Tuple, Callable, Optional
from zipfile import ZipFile

import numpy as np
import torch
from jsonargparse import CLI
from torch import Tensor

from comer.datamodules.crohme import extract_data_entries, DataEntry
from comer.datamodules.crohme.batch import build_batches_from_samples, BatchTuple, Batch
from comer.datamodules.crohme.variants.collate import collate_fn
from comer.modules import CoMERSupervised

checkpoint_path = "./bench/baseline_t112.ckpt"


def main(batch_size: int = 2, seeds: List[int] = None, gpu: int = 1,
         benches: Optional[List[str]] = None):
    if benches is None:
        benches = ["biggest", "smallest", "random"]
    if seeds is None:
        seeds = [1]
    print("init benchmarking")
    print("- seed: ", seeds)
    print("- batch_size: ", batch_size)
    print(f"- gpu: cuda:{gpu}")
    device = torch.device(f"cuda:{gpu}")

    with torch.no_grad():
        model = CoMERSupervised.load_from_checkpoint(checkpoint_path)
        model = model.eval().to(device)
        model.share_memory()

        print("model loaded")

        with ZipFile("data.zip") as archive:
            all_results: List[List[Tuple[str, float]]] = []
            if benches.count("smallest") > 0:
                all_results.append(run_benchmarks(
                    "Smallest",
                    get_sorted(extract_data_entries(archive, "train", to_device=device))[:batch_size],
                    batch_size,
                    model,
                    device
                ))

            if benches.count("biggest") > 0:
                all_results.append(run_benchmarks(
                    "Biggest",
                    get_sorted(extract_data_entries(archive, "train", to_device=device))[-batch_size:],
                    batch_size,
                    model,
                    device
                ))
            if benches.count("random") > 0:
                for seed in seeds:
                    all_results.append(run_benchmarks(
                        f"Random Seeded ({seed})",
                        get_random_seeded(extract_data_entries(archive, "train", to_device=device), device, batch_size,
                                          seed),
                        batch_size,
                        model,
                        device
                    ))

            print_results(all_results, batch_size=batch_size, seed=seed)


def run_benchmarks(
        name: str,
        entries: Tensor,
        batch_size: int,
        model,
        device: torch.device
) -> List[Tuple[str, float]]:
    torch.cuda.empty_cache()
    batches: List[BatchTuple] = build_batches_from_samples(
        entries,
        batch_size,
        is_labled=True,
        max_imagesize=int(32e6),
        batch_imagesize=int(32e6)
    )
    del entries

    shapes = [s.size() for s in batches[0][1]]
    batch: Batch = collate_fn(batches)
    batch.mask = batch.mask.to(device)
    batch.imgs = batch.imgs.to(device)

    del batches
    print("freeing mem...")
    torch.cuda.empty_cache()
    print("full warm up run...", end='', flush=True)
    full(batch, model, shapes)
    print("done")

    benches = [full, one_by_one_padded, one_by_one_unpadded]
    results: List[Tuple[str, float]] = []

    for bench in benches:
        print(f"Running bench \"{bench.__name__} ({name})\"...", end='', flush=True)
        start = time.time()
        bench(batch, model, shapes)
        results.append((f"{bench.__name__} ({name})", time.time() - start))
        print("done")
    print()
    del batch
    return results


def one_by_one_unpadded(batch: Batch, model, shapes):
    for i in range(batch.imgs.size(0)):
        shape = shapes[i]
        model.approximate_joint_search(
            batch.imgs[i:i + 1, :, :shape[1], :shape[2]],
            batch.mask[i:i + 1, :shape[1], :shape[2]]
        )


def one_by_one_padded(batch: Batch, model, shapes):
    for i in range(batch.imgs.size(0)):
        model.approximate_joint_search(batch.imgs[i:i + 1], batch.mask[i:i + 1])


def single_parallel_fn(model, imgs, mask):
    model.approximate_joint_search(imgs, mask)


def full(batch: Batch, model, shapes):
    model.approximate_joint_search(batch.imgs, batch.mask)


def fancy_print(name: str, time: float, ref: float):
    print(f"{name}"
          f"{time:.2f}s, {(time * 100 / ref):.2f}%,"
          f" (x{(ref / time):.2f})", )


def print_results(all_results: List[List[Tuple[str, float]]], seed, batch_size):
    if len(all_results) == 0:
        return
    bench_name_padding = 5

    max_len = max([max([len(res[0]) for res in results]) for results in all_results])

    header_line = "".join(["Bench Name".ljust(max_len + bench_name_padding),
                           "Total (s)".ljust(15), "Rel. Speed".ljust(15), "Speed-Up Factor"])

    print(f" Benchmark Results (batch_size: {batch_size}, seed: {seed}) ".center(len(header_line), "#"))
    print()
    print(header_line)
    print()

    for results in all_results:
        ref_time = results[0][1]

        for name, t in results:
            print(f"{name}".ljust(max_len + bench_name_padding), end='')
            print(f"{t:.2f}s".ljust(15), end='')
            print(f"{(t * 100 / ref_time):.2f}%".ljust(15), end='')
            print(f"x{(ref_time / t):.2f}")
        print()

    print()
    print("".ljust(len(header_line), "#"))


def get_random_seeded(entries, device: torch.device, batch_size: int, random_seed: int):
    # if requested, only return a subset of the dataset
    total_len = len(entries)
    idx_order = np.arange(total_len, dtype=int)
    np.random.seed(random_seed)
    np.random.shuffle(idx_order)
    idx_order = idx_order[:batch_size]
    return entries[idx_order]


def get_sorted(entries):
    get_entry_image_pixels: Callable[[DataEntry], int] = lambda x: x.image.size(1) * x.image.size(2)

    return (entries[
        np.argsort(
            np.vectorize(get_entry_image_pixels)(entries)
        )
    ])


if __name__ == '__main__':
    CLI(main)
