import time
from typing import List, Tuple, Callable, Optional
from zipfile import ZipFile

import numpy as np
import torch
from jsonargparse import CLI

from comer.datamodules.crohme import extract_data_entries, DataEntry, vocab
from comer.datamodules.crohme.batch import build_batches_from_samples, BatchTuple, Batch
from comer.datamodules.crohme.variants.collate import collate_fn
from comer.modules import CoMERSupervised

# checkpoint_path = "./bench/epoch3.ckpt"
checkpoint_path = "./bench/baseline_t112.ckpt"


def main(gpu: int = 1):
    print("init")
    print(f"- gpu: cuda:{gpu}")
    device = torch.device(f"cuda:{gpu}")

    with torch.no_grad():
        model = CoMERSupervised.load_from_checkpoint(checkpoint_path)
        model = model.eval().to(device)
        model.share_memory()

        print("model loaded")

        with ZipFile("data.zip") as archive:

            batch_tuple: BatchTuple = build_batches_from_samples(
                extract_data_entries(archive, "train", to_device=device),
                4,
                batch_imagesize=(2200 * 250 * 4),
                max_imagesize=(2200 * 250),
                is_labled=True,
                include_last_only_full=True
            )[-1]

            shapes = [s.size() for s in batch_tuple[1]]
            batch: Batch = collate_fn([batch_tuple]).to(device)

            torch.cuda.empty_cache()

            print("warm up")
            full(batch, model, shapes)
            n = 2
            print(f"benching normal {n} times")
            start_time = time.time()
            for _ in range(n):
                full(batch, model, shapes, use_new=False)
            print("total time: ", time.time() - start_time)

            print(f"benching new {n} times")
            start_time = time.time()
            for _ in range(n):
                full(batch, model, shapes, use_new=True)
            print("total time: ", time.time() - start_time)

def full(batch: Batch, model, shapes, use_new: bool = False):
    hyps = model.approximate_joint_search(batch.imgs, batch.mask, use_new=use_new)
    # print(hyps[0].score, len(hyps[0].seq), vocab.indices2words(hyps[0].seq))

if __name__ == '__main__':
    CLI(main)
