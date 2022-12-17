import math
from collections import defaultdict
from typing import List, Dict
from zipfile import ZipFile

import numpy as np
import torch
import torchvision.transforms as tr
from PIL.Image import Image
from jsonargparse import CLI
from pytorch_lightning import seed_everything
from torch import FloatTensor
from torchvision.transforms import ToPILImage

from comer.datamodules import Oracle
from comer.datamodules.crohme import extract_data_entries, vocab
from comer.datamodules.crohme.batch import build_batches_from_samples, BatchTuple, Batch, get_splitted_indices
from comer.datamodules.crohme.dataset import W_LO, H_LO, H_HI, W_HI
from comer.datamodules.crohme.variants.collate import collate_fn
from comer.datamodules.utils.randaug import RandAugment
from comer.datamodules.utils.transforms import ScaleToLimitRange
from comer.modules import CoMERSupervised
from comer.utils.utils import Hypothesis, to_bi_tgt_out
import torch.nn.functional as F

from Levenshtein import median, median_improve

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# checkpoint_path = "./bench/epoch3.ckpt"
checkpoint_path = './lightning_logs/version_48/checkpoints/ep=251-st=51982-valLoss=0.3578.ckpt'


def main(gpu: int = -1):
    print("init")
    print(f"- gpu: cuda:{gpu}")
    print(f"- cp: {checkpoint_path}")
    if gpu == -1:
        device = torch.device(f"cpu")
    else:
        device = torch.device(f"cuda:{gpu}")

    with torch.no_grad():
        # model: CoMERSupervised = CoMERSupervised.load_from_checkpoint(checkpoint_path)
        # model = model.eval().to(device)
        # model.share_memory()

        print("model loaded")

        with ZipFile("data.zip") as archive:

            seed_everything(7)

            full_train_data: 'np.ndarray[Any, np.dtype[DataEntry]]' = extract_data_entries(archive, "train", to_device=device)
            oracle = Oracle(full_train_data)

            th = math.log(0.995)

            all_hyps: Dict[str, Hypothesis] = torch.load("../hyps_st_15.pt", map_location=torch.device('cpu'))

            correct_hyps = 0
            correct_median = 0

            def calc_score(history: FloatTensor, tot_score: FloatTensor):
                summed_logits = torch.sum(history)
                min_logits = torch.min(history)
                return min_logits

            def calc_median(hyp: Hypothesis, fname: str):
                if (len(hyp.all_l2r_hyps) == 0) or (len(hyp.all_r2l_hyps) == 0):
                    return hyp.seq, hyp.history
                min_l2r = min((len(hyp.all_l2r_hyps), 2))
                min_r2l = min((len(hyp.all_r2l_hyps), 2))
                best_l2r_scores, best_l2r_idx = torch.topk(hyp.all_l2r_scores, k=min_l2r)
                best_r2l_scores, best_r2l_idx = torch.topk(hyp.all_r2l_scores, k=min_r2l)
                bstrs = []
                wlist = []
                abs_best_l2r = []
                abs_best_l2r_history = []
                abs_best_r2l = []
                abs_best_r2l_history = []
                for best, score in enumerate(best_l2r_scores):
                    if best == 0:
                        abs_best_l2r = hyp.all_l2r_hyps[best_l2r_idx[best]].tolist()
                        abs_best_l2r_history = hyp.all_l2r_history[best_l2r_idx[best]]
                    bstrs.append(bytes(hyp.all_l2r_hyps[best_l2r_idx[best]].tolist()))
                    wlist.append(1 / (100 * abs(float(score))))
                for best, score in enumerate(best_r2l_scores):
                    if best == 0:
                        abs_best_r2l = hyp.all_r2l_hyps[best_r2l_idx[best]].tolist()
                        abs_best_r2l_history = hyp.all_r2l_history[best_r2l_idx[best]]
                    bstrs.append(bytes(hyp.all_r2l_hyps[best_r2l_idx[best]].tolist()))
                    wlist.append(1 / (100 * abs(float(score))))
                if len(abs_best_l2r) == 0:
                    abs_best_l2r = abs_best_r2l
                    abs_best_l2r_history = abs_best_r2l_history
                if len(abs_best_r2l) == 0:
                    abs_best_r2l = abs_best_l2r
                    abs_best_r2l_history = abs_best_l2r_history

                mstr = list(bytearray(median(bstrs, wlist), "utf-8"))
                mhistory = hyp.history.copy()
                if len(mhistory) != len(mstr):
                    if len(mhistory) > len(mstr):
                        mhistory = mhistory[:len(mstr)]
                    else:
                        for i in range(abs(len(mhistory) - len(mstr))):
                            mhistory.append(hyp.score / 2)
                # mstr = hyp.seq
                mstr_len = len(mstr)

                gt = oracle.get_gt_indices(fname)
                gt_len = len(gt)

                for i, token in enumerate(abs_best_l2r):
                    if (i >= gt_len) or (i >= mstr_len) or (gt[i] != token):
                        break
                    mstr[i] = token
                    mhistory[i] = abs_best_l2r_history[i]

                i = 0
                for r2l_i, token in reversed(list(enumerate(abs_best_r2l))):
                    if (i >= gt_len) or (i >= mstr_len) or gt[gt_len - i - 1] != token:
                        break
                    mstr[mstr_len - 1 - i] = token
                    mhistory[mstr_len - 1 - i] = abs_best_r2l_history[r2l_i]
                    i += 1

                return mstr, mhistory

            counters = defaultdict(float)
            total_conf_correct = 0.0
            min_correct_conf = float('Inf')

            def calc_vec(hyp: Hypothesis):
                # token_vec = np.zeros(vocab.__len__() + 1)
                # total = len(hyp.all_r2l_hyps) + len(hyp.all_l2r_hyps)
                # token_vec[vocab.__len__()] = total
                # for idx, token in enumerate(hyp.seq):
                #     token_vec[token] += hyp.history[idx] * 100 * tota

                # token_vec = np.zeros(400)
                # for idx, token in enumerate(hyp.seq):
                #     token_vec[idx] = token
                #     token_vec[200 + idx] = hyp.history[idx]

                token_vec = np.zeros(vocab.__len__() * 2)
                total = len(hyp.all_r2l_hyps) + len(hyp.all_l2r_hyps)
                for idx, token in enumerate(hyp.seq):
                    token_vec[token] += hyp.history[idx] * 100 * total
                    token_vec[vocab.__len__() + token] += hyp.best_rev[idx] * 100 * total
                return token_vec

            X_normal = []
            y_normal = []

            for fname, hyp in all_hyps.items():
                lev_dist = oracle.levenshtein_indices(fname, hyp.seq)
                y = 0

                if lev_dist == 0:
                    y = 1
                    counters["correct"] += 1
                    min_conf = min(hyp.history)
                    total_conf_correct += min_conf
                    if min_conf < min_correct_conf:
                        min_correct_conf = min_conf



                mpred, mhistory = calc_median(hyp, fname)
                mpred_lev_dist = oracle.levenshtein_indices(fname, mpred)
                if mpred_lev_dist == 0:
                    counters["median_correct"] += 1

                if min(mhistory) >= th:
                    if np.random.random() < 1.2:
                        X_normal.append(calc_vec(hyp))
                        y_normal.append(y)
                    counters["median_conf_passed"] += 1
                    if mpred_lev_dist == 0:
                        counters["median_conf_passed_correct"] += 1
                    else:
                        counters["median_conf_lev_dist"] += mpred_lev_dist

                if min(hyp.history) >= th and min(hyp.best_rev) >= th:
                    counters["min_conf_passed"] += 1
                    if lev_dist == 0:
                        counters["min_conf_rev_score_correct"] += min(hyp.best_rev)
                        counters["min_conf_passed_correct"] += 1
                    else:
                        counters["min_conf_rev_score_incorrect"] += min(hyp.best_rev)
                        counters["min_conf_lev_dist"] += lev_dist

            print("Oracle", len(all_hyps), counters["correct"],
                  f"{counters['correct'] * 100 / len(all_hyps):.2f}",
                  f'{math.exp(total_conf_correct / counters["correct"])}',
                  math.exp(min_correct_conf)
                  )
            print(len(all_hyps), counters["median_correct"], f"{counters['median_correct'] * 100 / len(all_hyps):.2f}")
            print("MinConf", counters["min_conf_passed"],
                  f'{counters["min_conf_passed_correct"] * 100 / counters["min_conf_passed"]:.2f}',
                  counters["min_conf_lev_dist"] / (counters["min_conf_passed"] - counters["min_conf_passed_correct"]),
                  "Corr. AVG Min: ", f'{math.exp(counters["min_conf_rev_score_correct"] / counters["min_conf_passed_correct"]):.6f} '
                  "Incor. AVG Min: ", f'{math.exp(counters["min_conf_rev_score_incorrect"] / (counters["min_conf_passed"] - counters["min_conf_passed_correct"])):.6f} '
                  )
            print("MedianMinConf", counters["median_conf_passed"],
                  f'{counters["median_conf_passed_correct"] * 100 / counters["median_conf_passed"]:.2f}',
                  counters["median_conf_lev_dist"] / (counters["median_conf_passed"] - counters["median_conf_passed_correct"])
                  )


            clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
            clf.fit(X_normal, y_normal)

            for fname, hyp in all_hyps.items():
                if min(hyp.history) >= th and min(hyp.best_rev) >= th and clf.predict(calc_vec(hyp).reshape(1, -1)) == 1:
                    counters["svm_passed"] += 1
                    lev_dist = oracle.levenshtein_indices(fname, hyp.seq)
                    if lev_dist == 0:
                        counters["svm_passed_correct"] += 1
                    else:
                        counters["svm_lev_dist"] += lev_dist

            print("SVM", counters["svm_passed"],
                  f'{counters["svm_passed_correct"] * 100 / counters["svm_passed"]:.2f}',
                  counters["svm_lev_dist"] / (counters["svm_passed"] - counters["svm_passed_correct"])
                  )


            # def calc_min(th_pseudo_perc: float, exp: float = 1.0):
            #     th_min = math.log(th_pseudo_perc)
            #     min_conf_passed = 0
            #     min_conf_passed_correct = 0
            #     min_conf_lev_sum = 0
            #     correct_hyps = 0
            #
            #     for fname, hyp in all_hyps.items():
            #         lev_dist = oracle.levenshtein_indices(fname, hyp.seq)
            #         if lev_dist == 0:
            #             correct_hyps += 1
            #
            #         seq_len = len(hyp.seq)
            #         if (hyp.score / 2) * (exp ** (1 + 5 * math.log(seq_len))) > th_min:
            #             min_conf_passed += 1
            #             if lev_dist == 0:
            #                 min_conf_passed_correct += 1
            #             else:
            #                 min_conf_lev_sum += lev_dist
            #         # if len(hyp.history) > 0:
            #         #     min_score = min(hyp.history)
            #         #     if (min_score * (exp ** (1 + 5 * math.log(seq_len)))) > th_min:
            #         #         # if (min_score) > th_min:
            #         #         min_conf_passed += 1
            #         #         if lev_dist == 0:
            #         #             min_conf_passed_correct += 1
            #         #         else:
            #         #             min_conf_lev_sum += lev_dist
            #     return min_conf_passed, min_conf_passed_correct,\
            #         min_conf_passed_correct / correct_hyps if correct_hyps > 0 else 0,\
            #         min_conf_passed_correct / min_conf_passed  if min_conf_passed > 0 else 0
            #
            #
            # curr_min = 0.95
            # curr_max = 1.0
            #
            # steps = 1000
            # step_size = (curr_max - curr_min) / steps
            #
            # best = 0.95
            # best_score = float('-Inf')
            # best_cov, best_pass, best_corr, best_pct, best_exp = 0, 0, 0, 0, 0
            #
            # cov_exp = 15
            #
            # exp_steps = 20
            # exp_min = 1.0
            # exp_max = 1.2
            # exp_step_size = (exp_max - exp_min) / exp_steps

            # for i in range(steps):
            #     curr = curr_min + step_size * i
            #     for x in range(exp_steps):
            #         exp = exp_min + x * exp_step_size
            #         m_pass, m_corr, cov_pct, correct_pct = calc_min(curr, exp)
            #
            #         # print(curr, cov_pct, correct_pct, (cov_pct * correct_pct))
            #
            #         if (m_corr * (correct_pct ** cov_exp)) > best_score:
            #             best = curr
            #             best_score = (m_corr * (correct_pct ** cov_exp))
            #             best_cov = cov_pct
            #             best_pass = m_pass
            #             best_corr = m_corr
            #             best_pct = correct_pct
            #             best_exp = exp
            # print(best, best_exp, best_score, best_cov, best_pass, best_corr, best_pct)
            # print(calc_min(0.9875))


            # min = 0.95
            # change = 1

            # print("Hyps", len(all_hyps), "Correct", correct_hyps)
            # print("AVG", "Passed: ", avg_conf_passed, " Correct: ", avg_conf_passed_correct, f"{avg_conf_passed_correct * 100 / avg_conf_passed:.2f}", avg_conf_lev_sum / (avg_conf_passed - avg_conf_passed_correct))
            # print("MIN", "Passed: ", min_conf_passed, " Correct: ", min_conf_passed_correct, f"{min_conf_passed_correct * 100 / min_conf_passed:.2f}", min_conf_lev_sum / (min_conf_passed - min_conf_passed_correct))
            #

            # labeled_indices, unlabeled_indices = get_splitted_indices(
            #     full_train_data,
            #     unlabeled_pct=0.85,
            #     sorting_mode=1
            # )
            # labeled_data, unlabeled_data = full_train_data[labeled_indices], full_train_data[
            #     unlabeled_indices]
            #
            # pseudo_labeling_batches = build_batches_from_samples(
            #     unlabeled_data,
            #     4
            # )
            #
            # all_hyps = {}
            #
            # for batch_raw in pseudo_labeling_batches:
            #     batch = collate_fn([batch_raw]).to(device=device)
            #     hyps: List[Hypothesis] = model.approximate_joint_search(batch.imgs, batch.mask, use_new=True, debug=False)
            #     for i, hyp in enumerate(hyps):
            #         all_hyps[batch.img_bases[i]] = hyp
            #
            # torch.save(all_hyps, "hyps_st_15.pt")


            # entries = extract_data_entries(archive, "2014", to_device=device)
            #
            # batch_tuple: List[BatchTuple] = build_batches_from_samples(
            #     entries,
            #     4,
            #     batch_imagesize=(2200 * 250 * 4),
            #     max_imagesize=(2200 * 250),
            #     include_last_only_full=True
            # )[::-1]
            #
            # batch: Batch = collate_fn([batch_tuple[0]]).to(device)

            # batch_tuple_single: List[BatchTuple] = build_batches_from_samples(
            #     entries,
            #     1,
            #     batch_imagesize=(2200 * 250 * 4),
            #     max_imagesize=(2200 * 250),
            #     include_last_only_full=True
            # )[::-1]
            # test: BatchTuple = batch_tuple_single[2]
            # test[0].append(batch_tuple_single[3][0][0])
            # test[1].append(batch_tuple_single[3][1][0])
            # test[2].append(batch_tuple_single[3][2][0])
            # batch_single: Batch = collate_fn([test]).to(device)

            # feature, mask = model.comer_model.encoder(batch_single.imgs, batch_single.mask)
            # corner_vecs = torch.cat((
            #     feature[1, 0, 0, :],
            #     feature[1, 11, 0, :],
            #     feature[1, 0, 70, :],
            #     feature[1, 11, 70, :]
            # ))
            # corner_vecs_m = torch.tensor([
            #     mask[1, 0, 0],
            #     mask[1, 11, 0],
            #     mask[1, 0, 70],
            #     mask[1, 11, 70]
            # ])
            #
            # np.savetxt("test_1_ft.txt", corner_vecs.cpu().numpy())
            # np.savetxt("test_1_mask.txt", corner_vecs_m.cpu().numpy())
            #
            # feature, mask = model.comer_model.encoder(batch.imgs, batch.mask)
            # corner_vecs = torch.cat((
            #     feature[1, 0, 0, :],
            #     feature[1, 11, 0, :],
            #     feature[1, 0, 70, :],
            #     feature[1, 11, 70, :]
            # ))
            # corner_vecs_m = torch.tensor([
            #     mask[1, 0, 0],
            #     mask[1, 11, 0],
            #     mask[1, 0, 70],
            #     mask[1, 11, 70]
            # ])
            #
            # np.savetxt("test_2_ft.txt", corner_vecs.cpu().numpy())
            # np.savetxt("test_2_mask.txt", corner_vecs_m.cpu().numpy())

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
            #             single_logit = hyps[i].history[sym]
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
