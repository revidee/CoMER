import itertools
import math
import os
from typing import Tuple, List
from zipfile import ZipFile

import numpy as np
import torch

from comer.datamodules import Oracle
from comer.datamodules.crohme import extract_data_entries, get_splitted_indices, build_batches_from_samples, BatchTuple
from comer.datamodules.crohme.variants.collate import collate_fn
from comer.modules import CoMERFixMatchInterleavedLogitNormTempScale
from comer.utils import ECELoss
from comer.utils.conf_measures import score_avg, score_sum, score_bimin
from comer.utils.utils import Hypothesis

def gpu_str(gpu: int):
    return f"cuda:{gpu}" if gpu >= 0 else "cpu"
def main(gpu: int = 0):
    print(f"- gpu: {gpu_str(gpu)}")
    device = torch.device(gpu_str(gpu))
    with ZipFile("data.zip") as archive:
        # TODO: Modify to your needs
        cps = [
            # ("./lightning_logs/version_64/checkpoints/optimized_ts_0.5146.ckpt", "t0_02_opt"),
            # ("./lightning_logs/version_70/checkpoints/optimized_ts_0.5405.ckpt", "t0_04_opt"),
            # ("./lightning_logs/version_65/checkpoints/optimized_ts_0.5505.ckpt", "t0_05_opt"),
            ("./lightning_logs/version_71/checkpoints/epoch=197-step=52074-val_ExpRate=0.5321.ckpt", "t0_0625"),
            # ("./lightning_logs/version_71/checkpoints/optimized_ts_0.5338.ckpt", "t0_0625_opt"),
            # ("./lightning_logs/version_69/checkpoints/optimized_ts_0.556297.ckpt", "t0_075_opt"),
            # ("./lightning_logs/version_66/checkpoints/optimized_ts_0.5421.ckpt", "t0_1_opt"),
            # ("./lightning_logs/version_67/checkpoints/optimized_ts_0.5038.ckpt", "t0_2_opt"),
            # ("./lightning_logs/version_68/checkpoints/optimized_ts_0.4829.ckpt", "t0_5_opt"),
            # ("./lightning_logs/version_25/checkpoints/epoch=293-step=154644-val_ExpRate=0.5488.ckpt", "original"),
        ]
        oracle, test_batches, pseudo_labeling_batches = get_testable_sets(archive, device)

        # None = use the learned & saved temperature of the model, float = use the given temperature instead
        to_test_temps = [1, 3]

        # 0: suffix used to save the hyp predictions
        # 1: data set which to make predictions for
        to_test_data_sets: List[Tuple[str, List[BatchTuple]]] = [
            ("_test", test_batches),
            ("", pseudo_labeling_batches),
        ]

        ece = ECELoss()

        def hyp_to_triplet_ori(fname_and_hyp: Tuple[str, Hypothesis]):
            fname, hyp = fname_and_hyp
            hyp_len = len(hyp.seq)
            return (math.exp(score_avg(hyp)) if hyp_len > 0 else 0, hyp.seq, oracle.get_gt_indices(fname))
        def hyp_to_triplet_avg(fname_and_hyp: Tuple[str, Hypothesis]):
            fname, hyp = fname_and_hyp
            hyp_len = len(hyp.seq)
            return (math.exp(score_sum(hyp) / hyp_len) if hyp_len > 0 else 0, hyp.seq, oracle.get_gt_indices(fname))

        def hyp_to_triplet_bimin(fname_and_hyp: Tuple[str, Hypothesis]):
            fname, hyp = fname_and_hyp
            hyp_len = len(hyp.seq)
            return (math.exp(score_bimin(hyp)) if hyp_len > 0 else 0, hyp.seq, oracle.get_gt_indices(fname))

        scoring_fns = [
            ("original", hyp_to_triplet_ori),
            ("sum", hyp_to_triplet_avg),
            ("bimin", hyp_to_triplet_bimin),
        ]


        scoring_fn_names = [name for (name, _) in scoring_fns]
        with torch.inference_mode():
            for (cp, name) in cps:
                model = None
                torch.cuda.empty_cache()
                all_hyps = {}

                print(f"Loading {cp}...")
                model: CoMERFixMatchInterleavedLogitNormTempScale \
                    = CoMERFixMatchInterleavedLogitNormTempScale.load_from_checkpoint(
                    cp
                )
                model = model.eval().to(device)
                model.share_memory()
                print("model loaded")


                for (save_ds_suffix, set) in to_test_data_sets:
                    for temp in to_test_temps:
                        save_path = f"./hyps_s_35_{name}{f'_{temp}' if temp is not None else ''}{save_ds_suffix}.pt"
                        exists = os.path.exists(save_path)
                        if not exists:
                            ten_pct_steps = np.floor(np.linspace(0, len(set), 10, endpoint=False))
                            print(f"LN {name}{save_ds_suffix}, progress: ", end="", flush=True)
                            progress = 0
                            all_hyps[save_path] = {}
                            all_hyps_save_path = all_hyps[save_path]
                            for i, batch_raw in enumerate(set):
                                if i in ten_pct_steps:
                                    print(progress, end="", flush=True)
                                    progress += 1
                                batch = collate_fn([batch_raw]).to(device=device)
                                hyps = model.approximate_joint_search(
                                    batch.imgs, batch.mask, use_new=True, debug=False, save_logits=False, temperature=temp
                                )
                                for i, hyp in enumerate(hyps):
                                    all_hyps_save_path[batch.img_bases[i]] = hyp
                            torch.save(all_hyps_save_path, save_path)
                            print(" saved")
                        else:
                            all_hyps[save_path] = torch.load(save_path, map_location=device)
                            print(f"LN {name} loaded from save")
                # do the actual eval
                for (save_ds_suffix, _) in to_test_data_sets:
                    pct_slots = list(itertools.repeat("", len(to_test_temps) + len(scoring_fns) * len(to_test_temps)))

                    for temp_idx, temp in enumerate(to_test_temps):
                        save_path = f"./hyps_s_35_{name}{f'_{temp}' if temp is not None else ''}{save_ds_suffix}.pt"
                        for i, (_, tf) in enumerate(scoring_fns):
                            ece_score, acc = ece.ece_for_predictions(map(tf, all_hyps[save_path].items()))
                            pct_slots[temp_idx] = f"{acc * 100:.2f}"
                            pct_slots[len(to_test_temps) + i + temp_idx * len(scoring_fns)] = f"{ece_score * 100:.2f}"
                    curr_temp = 1.0
                    if hasattr(model, "current_temperature"):
                        curr_temp = model.current_temperature.item()
                    print(f"{name}{save_ds_suffix}", f"ts: {curr_temp}, temps: {to_test_temps}, confs: {scoring_fn_names}")
                    print("\t".join(pct_slots[:len(to_test_temps)]), end="")
                    print("\t\t", end="")
                    for i in range(len(to_test_temps)):
                        print("\t".join(pct_slots[len(to_test_temps)+i*len(scoring_fns):len(to_test_temps)+(i+1)*len(scoring_fns)]), end="")
                        print("\t\t", end="")
                    print()

def get_testable_sets(archive: ZipFile, device : torch.device):
    full_data: 'np.ndarray[Any, np.dtype[DataEntry]]' = extract_data_entries(archive, "train",
                                                                             to_device=device)
    full_data_test: 'np.ndarray[Any, np.dtype[DataEntry]]' = extract_data_entries(archive, "2019",
                                                                                  to_device=device)
    oracle = Oracle(full_data)
    oracle.add_data(full_data_test)

    labeled_indices, unlabeled_indices = get_splitted_indices(
        full_data,
        unlabeled_pct=0.65,
        sorting_mode=1
    )
    labeled_data, unlabeled_data = full_data[labeled_indices], full_data[
        unlabeled_indices]

    pseudo_labeling_batches = build_batches_from_samples(
        unlabeled_data,
        4
    )
    test_batches = build_batches_from_samples(
        full_data_test,
        4
    )
    return oracle, test_batches, pseudo_labeling_batches