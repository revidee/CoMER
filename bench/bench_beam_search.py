import time
from typing import List, Tuple, Callable, Optional
from zipfile import ZipFile

import numpy as np
import torch
from jsonargparse import CLI

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

            batches: List[BatchTuple] = build_batches_from_samples(
                extract_data_entries(archive, "train", to_device=device),
                batch_size,
                batch_imagesize=(2200 * 250 * 4),
                max_imagesize=(2200 * 250),
                is_labled=True,
                include_last_only_full=True
            )

            all_results: List[List[Tuple[str, float]]] = []
            suites = generate_bench_suites(batches, benches, seeds)
            del batches

            for suite in suites:
                all_results.append(run_benchmarks(
                    suite[0],
                    suite[1],
                    model,
                    device
                ))

            print_results(all_results, batch_size=batch_size, seeds=seeds)


def generate_bench_suites(sorted_batches: List[BatchTuple],
                          benches: List[str], seeds: List[int]) -> List[Tuple[str, BatchTuple]]:
    suites: List[Tuple[str, BatchTuple]] = []

    if benches.count("smallest") > 0:
        suites.append(("Smallest", sorted_batches[0]))

    if benches.count("biggest") > 0:
        suites.append(("Biggest", sorted_batches[-1]))

    if benches.count("random") > 0:
        batch_len = len(sorted_batches)
        for seed in seeds:
            np.random.seed(seed)
            suites.append((f"Random Seeded ({seed})", sorted_batches[np.random.randint(0, batch_len)]))

    return suites


def run_benchmarks(
        name: str,
        batch_tuple: BatchTuple,
        model,
        device: torch.device
) -> List[Tuple[str, float]]:
    torch.cuda.empty_cache()

    shapes = [s.size() for s in batch_tuple[1]]
    batch: Batch = collate_fn([batch_tuple]).to(device)

    print(f"Run Benchmark Suite: {name}, img_size: {batch.imgs.size()}")
    print(f"freeing mem...")
    torch.cuda.empty_cache()
    print("full warm up run...", end='', flush=True)
    full(batch, model, shapes)
    print("done")

    benches = [full, one_by_one_padded, two_by_two_padded, one_by_one_unpadded]
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

def two_by_two_padded(batch: Batch, model, shapes):
    for i in range(batch.imgs.size(0) // 2):
        model.approximate_joint_search(batch.imgs[(2 * i):(2 * (i + 1))], batch.mask[(2 * i):(2 * (i + 1))])


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


def print_results(all_results: List[List[Tuple[str, float]]], seeds: List[int], batch_size):
    if len(all_results) == 0:
        return
    bench_name_padding = 5

    max_len = max([max([len(res[0]) for res in results]) for results in all_results])

    header_line = "".join(["Bench Name".ljust(max_len + bench_name_padding),
                           "Total (s)".ljust(15), "Rel. Speed".ljust(15), "Speed-Up Factor"])

    title_line = f" Benchmark Results (batch_size: {batch_size}, seed: {seeds}) "
    max_header_title = max([len(title_line), len(header_line)])
    print(title_line.center(max_header_title, "#"))
    print()
    print(header_line)
    print()

    total_relatives = []
    total_times = []

    for results in all_results:
        ref_time = results[0][1]

        total_relatives.append([(ref_time / res[1]) for res in results])
        total_times.append([res[1] for res in results])

        for name, t in results:
            print(f"{name}".ljust(max_len + bench_name_padding), end='')
            print(f"{t:.2f}s".ljust(15), end='')
            print(f"{(t * 100 / ref_time):.2f}%".ljust(15), end='')
            print(f"x{(ref_time / t):.2f}")
        print()
    print("-".ljust(max_header_title, "-"))
    print()
    print("Aggregates")
    print()

    stacked_rels = np.vstack(total_relatives)
    stacked_times = np.vstack(total_times)

    means = np.mean(stacked_rels, axis=0)
    stds = np.std(stacked_rels, axis=0)
    total_sum = np.sum(stacked_times, axis=0)

    for i in range(len(all_results[0])):
        base_name = all_results[0][i][0].split(" ")[0]
        print(f"{base_name}".ljust(max_len + bench_name_padding), end='')
        print(f"{total_sum[i]:.2f}s".ljust(15), end='')
        print(f"{(total_sum[i] * 100 / total_sum[0]):.2f}%".ljust(15), end='')
        print(f"x{(total_sum[0] / total_sum[i]):.2f}".ljust(10), end='')
        print(f"x{means[i]:.2f} +- {stds[i]:.2f}".ljust(15))
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
