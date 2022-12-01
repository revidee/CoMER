from typing import List
from zipfile import ZipFile

import torch
import torchvision.transforms as tr
from PIL.Image import Image
from jsonargparse import CLI
from torchvision.transforms import ToPILImage

from comer.datamodules.crohme import extract_data_entries
from comer.datamodules.crohme.batch import build_batches_from_samples, BatchTuple, Batch
from comer.datamodules.crohme.dataset import W_LO, H_LO, H_HI, W_HI
from comer.datamodules.utils.randaug import RandAugment
from comer.datamodules.utils.transforms import ScaleToLimitRange

checkpoint_path = "./bench/epoch3.ckpt"
# checkpoint_path = "./bench/baseline_t112.ckpt"


def main(gpu: int = 1):
    print("init")
    print(f"- gpu: cuda:{gpu}")
    print(f"- cp: {checkpoint_path}")
    if gpu == -1:
        device = torch.device(f"cpu")
    else:
        device = torch.device(f"cuda:{gpu}")

    with torch.no_grad():
        # model = CoMERSupervised.load_from_checkpoint(checkpoint_path)
        # model = model.eval().to(device)
        # model.share_memory()
        #
        # print("model loaded")

        with ZipFile("data.zip") as archive:

            batch_tuple: List[BatchTuple] = build_batches_from_samples(
                extract_data_entries(archive, "2014"),
                4,
                batch_imagesize=(2200 * 250 * 4),
                max_imagesize=(2200 * 250),
                is_labled=True,
                include_last_only_full=True
            )[::-1]

            trans_list = [ToPILImage(), RandAugment(3), ScaleToLimitRange(w_lo=W_LO, w_hi=W_HI, h_lo=H_LO, h_hi=H_HI)]
            transform = tr.Compose(trans_list)

            file_names, images, labels, is_labled, src_idx = batch_tuple[0]

            transformed_ims = [transform(im) for im in images]

            Image.show(ToPILImage()(transformed_ims[0]))
            Image.show(ToPILImage()(transformed_ims[1]))
            Image.show(ToPILImage()(transformed_ims[2]))
            Image.show(ToPILImage()(transformed_ims[3]))


            # start_time = time.time()
            # for idx, batch_tup in enumerate(batch_tuple):
            #     if idx > 0:
            #         break
            #     batch: Batch = collate_fn([batch_tup]).to(device)
            #     hyps = model.approximate_joint_search(batch.imgs, batch.mask, use_new=True, debug=False)
            #     for i, hyp in enumerate(hyps):
            #         print(batch.img_bases[i], vocab.indices2words(hyp.seq))
                # hyps = model.approximate_joint_search(batch.imgs, batch.mask, use_new=False)
                # hyps_new = model.approximate_joint_search(batch.imgs, batch.mask, use_new=True)
                # for i, hyp_old in enumerate(hyps):
                #     if hyp_old.seq != hyps_new[i].seq:
                #         print("OLD:")
                #         model.approximate_joint_search(batch.imgs, batch.mask, use_new=False, debug=True)
                #         print("")
                #         print("")
                #         print("NEW:")
                #         model.approximate_joint_search(batch.imgs, batch.mask, use_new=True, debug=True)
                #         print("mismatch", batch.img_bases[0], idx)
                #         print("old: ", vocab.indices2words(hyp_old.seq))
                #         print("new: ", vocab.indices2words(hyps_new[i].seq))
                #         exit(1)
            # prof.export_stacks("profiler_stacks_gpu.txt", "self_cuda_time_total")
            # prof.export_stacks("profiler_stacks_cpu.txt", "self_cpu_time_total")
            # torch.save(prof.key_averages().table(sort_by="self_cpu_time_total"), "profiler_table.txt")
            # prof.export_chrome_trace("profiler_chrome.json")
            # print("total: ", time.time() - start_time)


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
