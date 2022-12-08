from typing import List
from zipfile import ZipFile

import numpy as np
import torch
import torchvision.transforms as tr
from PIL.Image import Image
from jsonargparse import CLI
from torchvision.transforms import ToPILImage

from comer.datamodules import Oracle
from comer.datamodules.crohme import extract_data_entries, vocab
from comer.datamodules.crohme.batch import build_batches_from_samples, BatchTuple, Batch
from comer.datamodules.crohme.dataset import W_LO, H_LO, H_HI, W_HI
from comer.datamodules.crohme.variants.collate import collate_fn
from comer.datamodules.utils.randaug import RandAugment
from comer.datamodules.utils.transforms import ScaleToLimitRange
from comer.modules import CoMERSupervised
from comer.utils.utils import Hypothesis, to_bi_tgt_out
import torch.nn.functional as F

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
        model: CoMERSupervised = CoMERSupervised.load_from_checkpoint(checkpoint_path)
        model = model.eval().to(device)
        model.share_memory()

        print("model loaded")

        with ZipFile("data.zip") as archive:
            entries = extract_data_entries(archive, "2014", to_device=device)
            oracle = Oracle(entries)
            batch_tuple: List[BatchTuple] = build_batches_from_samples(
                entries,
                4,
                batch_imagesize=(2200 * 250 * 4),
                max_imagesize=(2200 * 250),
                include_last_only_full=True
            )[::-1]

            batch: Batch = collate_fn([batch_tuple[0]]).to(device)

            batch_tuple_single: List[BatchTuple] = build_batches_from_samples(
                entries,
                1,
                batch_imagesize=(2200 * 250 * 4),
                max_imagesize=(2200 * 250),
                include_last_only_full=True
            )[::-1]
            test: BatchTuple = batch_tuple_single[2]
            test[0].append(batch_tuple_single[3][0][0])
            test[1].append(batch_tuple_single[3][1][0])
            test[2].append(batch_tuple_single[3][2][0])
            batch_single: Batch = collate_fn([test]).to(device)

            feature, mask = model.comer_model.encoder(batch_single.imgs, batch_single.mask)
            corner_vecs = torch.cat((
                feature[1, 0, 0, :],
                feature[1, 11, 0, :],
                feature[1, 0, 70, :],
                feature[1, 11, 70, :]
            ))
            corner_vecs_m = torch.tensor([
                mask[1, 0, 0],
                mask[1, 11, 0],
                mask[1, 0, 70],
                mask[1, 11, 70]
            ])

            np.savetxt("test_1_ft.txt", corner_vecs.cpu().numpy())
            np.savetxt("test_1_mask.txt", corner_vecs_m.cpu().numpy())

            feature, mask = model.comer_model.encoder(batch.imgs, batch.mask)
            corner_vecs = torch.cat((
                feature[1, 0, 0, :],
                feature[1, 11, 0, :],
                feature[1, 0, 70, :],
                feature[1, 11, 70, :]
            ))
            corner_vecs_m = torch.tensor([
                mask[1, 0, 0],
                mask[1, 11, 0],
                mask[1, 0, 70],
                mask[1, 11, 70]
            ])

            np.savetxt("test_2_ft.txt", corner_vecs.cpu().numpy())
            np.savetxt("test_2_mask.txt", corner_vecs_m.cpu().numpy())

            # hyps: List[Hypothesis] = model.approximate_joint_search(batch.imgs, batch.mask, use_new=True, debug=False)
            #
            # score_batch = Batch(batch.img_bases, batch.imgs, batch.mask,[hyp.seq for hyp in hyps], 0, 0, 4).to(device)
            #
            # tgt, out = to_bi_tgt_out(score_batch.labels, device)
            # logits = F.log_softmax(
            #     model(score_batch.imgs, score_batch.mask, tgt),
            #     dim=-1
            # )
            #
            # batch_len = len(score_batch)
            #
            # for i, fname in enumerate(batch.img_bases):
            #     print(f"{fname}: (hyp l2r: {hyps[i].was_l2r})")
            #     print(f"{hyps[i].score:.4f}".ljust(5), f"{oracle.confidence_indices(fname, hyps[i].seq):.4f}")
            #     gt = batch.labels[i]
            #     gt_len = len(gt)
            #     pred = hyps[i].seq
            #     pred_len = len(pred)
            #     lines = [[], [], [], [], [], [], [], []]
            #     prev_logit = 0
            #     diff = gt_len - pred_len
            #     for sym in range(gt_len):
            #         str_gt_l2r = vocab.idx2word[gt[sym]]
            #         str_pred_l2r = ""
            #         str_pred_r2l = ""
            #         logit_l2r_str = ""
            #         logit_r2l_str = ""
            #         logit_str = ""
            #         logit_r2l_str_gt_sym = ""
            #
            #         if sym < pred_len:
            #             str_pred_l2r = vocab.idx2word[pred[sym]]
            #             logit_l2r_str = f"{logits[i, sym, pred[sym]]:.1E}"
            #             if hyps[i].was_l2r:
            #                 single_logit = hyps[i].history[sym] - prev_logit
            #                 prev_logit = hyps[i].history[sym]
            #             else:
            #                 if sym < pred_len - 1:
            #                     single_logit = hyps[i].history[sym] - hyps[i].history[sym + 1]
            #                 else:
            #                     single_logit = hyps[i].history[sym]
            #
            #             logit_str = f"{single_logit:.1E}"
            #         if sym >= diff:
            #             str_pred_r2l = vocab.idx2word[pred[sym - diff]]
            #             logit_r2l_str = f"{logits[i + batch_len, pred_len - sym + diff - 1, pred[sym - diff]]:.1E}"
            #             logit_r2l_str_gt_sym = f"{logits[i + batch_len, pred_len - sym + diff - 1, gt[sym]]:.1E}"
            #
            #
            #         maxlen = max((len(str_gt_l2r), len(str_pred_l2r),
            #                       len(str_pred_r2l), len(logit_str), len(logit_l2r_str),
            #                       len(logit_r2l_str), len(logit_r2l_str_gt_sym)))
            #         lines[0].append(str_gt_l2r.center(maxlen + 1))
            #         lines[1].append(str_pred_l2r.center(maxlen + 1))
            #         lines[2].append(logit_str.ljust(maxlen + 1))
            #         lines[3].append(logit_l2r_str.ljust(maxlen + 1))
            #
            #         lines[4].append(str_gt_l2r.center(maxlen + 1))
            #         lines[5].append(str_pred_r2l.center(maxlen + 1))
            #         lines[6].append(logit_r2l_str.ljust(maxlen + 1))
            #         lines[7].append(logit_r2l_str_gt_sym.ljust(maxlen + 1))
            #     print("".join(lines[0]))
            #     print("".join(lines[1]))
            #     print("".join(lines[2]))
            #     print("".join(lines[3]))
            #     print()
            #     print("".join(lines[4]))
            #     print("".join(lines[5]))
            #     print("".join(lines[2]))
            #     print("".join(lines[6]))
            #     print("".join(lines[7]))
            #     print(f"len gt: {len(batch.labels[i])} vs. pred: {len(hyps[i].seq)}")

            # trans_list = [ToPILImage(), RandAugment(3), ScaleToLimitRange(w_lo=W_LO, w_hi=W_HI, h_lo=H_LO, h_hi=H_HI)]
            # transform = tr.Compose(trans_list)
            #
            # file_names, images, labels, is_labled, src_idx = batch_tuple[0]
            #
            # transformed_ims = [transform(im) for im in images]
            #
            # Image.show(ToPILImage()(transformed_ims[0]))
            # Image.show(ToPILImage()(transformed_ims[1]))
            # Image.show(ToPILImage()(transformed_ims[2]))
            # Image.show(ToPILImage()(transformed_ims[3]))


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
