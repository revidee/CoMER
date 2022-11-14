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
    print(f"- cp: {checkpoint_path}")
    if gpu == -1:
        device = torch.device(f"cpu")
    else:
        device = torch.device(f"cuda:{gpu}")

    with torch.no_grad():
        model = CoMERSupervised.load_from_checkpoint(checkpoint_path)
        model = model.eval().to(device)
        model.share_memory()

        print("model loaded")

        with ZipFile("data.zip") as archive:

            batch_tuple: List[BatchTuple] = build_batches_from_samples(
                extract_data_entries(archive, "2014", to_device=device),
                1,
                batch_imagesize=(2200 * 250 * 4),
                max_imagesize=(2200 * 250),
                is_labled=True,
                include_last_only_full=True
            )

            for idx, batch_tup in enumerate(batch_tuple):
                batch: Batch = collate_fn([batch_tup]).to(device)
                hyps = model.approximate_joint_search(batch.imgs, batch.mask, use_new=False)
                hyps_new = model.approximate_joint_search(batch.imgs, batch.mask, use_new=True)
                for i, hyp_old in enumerate(hyps):
                    if hyp_old.seq != hyps_new[i].seq:
                        print("OLD:")
                        model.approximate_joint_search(batch.imgs, batch.mask, use_new=False, debug=True)
                        print("")
                        print("")
                        print("NEW:")
                        model.approximate_joint_search(batch.imgs, batch.mask, use_new=True, debug=True)
                        print("mismatch", batch.img_bases[0], idx)
                        print("old: ", vocab.indices2words(hyp_old.seq))
                        print("new: ", vocab.indices2words(hyps_new[i].seq))
                        exit(1)

            print(batch.img_bases)

            # n = 2
            # print(f"benching normal {n} times")
            # start_time = time.time()
            # for _ in range(n):
            #     full(batch, model, shapes, use_new=False)
            # print("total time: ", time.time() - start_time)
            #
            # print(f"benching new {n} times")
            # start_time = time.time()
            # for _ in range(n):
            #     full(batch, model, shapes, use_new=True)
            # print("total time: ", time.time() - start_time)

def full(batch: Batch, model, shapes, use_new: bool = False):
    return model.approximate_joint_search(batch.imgs, batch.mask, use_new=use_new)
    # print(hyps[0].score, len(hyps[0].seq), vocab.indices2words(hyps[0].seq))

if __name__ == '__main__':
    CLI(main)
