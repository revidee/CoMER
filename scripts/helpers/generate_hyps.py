from __future__ import annotations
import re
from typing import List, Union, Tuple, Any
from zipfile import ZipFile

import torch
from jsonargparse import CLI
from pytorch_lightning import seed_everything

from comer.datamodules.crohme import extract_data_entries, build_batches_from_samples, BatchTuple
from model_lookups import AVAILABLE_MODELS


def generate_hyps(checkpoints: List[Tuple[str, Any, str]],
                  datasets_with_suffix: List[Tuple[str, List[BatchTuple]]],
                  device: torch.device,
                  temps: List[Union[float, None]] = None,
                  output_root: str = "./"
                  ):
    if temps is None:
        temps = [None]
    with torch.inference_mode():
        for (cp, model_class, name) in checkpoints:
            model = None
            torch.cuda.empty_cache()
            all_hyps = {}

            print(f"Loading {cp}...")
            model: model_class \
                = model_class.load_from_checkpoint(
                cp
            )
            model = model.eval().to(device)
            model.share_memory()
            print("model loaded")

            for (save_ds_suffix, set) in datasets_with_suffix:
                for temp in temps:
                    save_path = f"{output_root}{name}{f'_{temp}' if temp is not None else ''}{save_ds_suffix}.pt"
                    exists = os.path.exists(save_path)
                    if not exists:
                        ten_pct_steps = np.floor(np.linspace(0, len(set), 10, endpoint=False))
                        print(f"LN {name}{save_ds_suffix}, progress: ", end="", flush=True)
                        progress = 0
                        all_hyps = {}
                        for i, batch_raw in enumerate(set):
                            if i in ten_pct_steps:
                                print(progress, end="", flush=True)
                                progress += 1
                            batch = collate_fn([batch_raw]).to(device=device)
                            hyps = model.approximate_joint_search(
                                batch.imgs, batch.mask, use_new=True, debug=False, save_logits=False, temperature=temp,
                                global_pruning='none'
                            )
                            for i, hyp in enumerate(hyps):
                                all_hyps[batch.img_bases[i]] = hyp
                        torch.save(all_hyps, save_path)
                        print(" saved")


def main(gpu: int, cps: List[str], models: List[str], years: List[int], temps=None, batch_size: int = 4):
    if temps is None:
        temps = [1]
    with ZipFile("data.zip") as archive:
        seed_everything(7)
        device = torch.device(f"cuda:{gpu}")
        sets = []

        for year in years:
            full_data_test: 'np.ndarray[Any, np.dtype[DataEntry]]' = extract_data_entries(archive, f"{year}",
                                                                                          to_device=device)

            test_batches = build_batches_from_samples(
                full_data_test,
                batch_size
            )
            sets.append((f"_test{year}", test_batches))

        model_list = []
        for model, cp in zip(models, cps):
            if model not in AVAILABLE_MODELS:
                continue
            cp: str
            VERSION_STR = "version_"
            match = re.search("^version_%d$", cp)
            if match is None:
                continue

            version_start = cp.find(VERSION_STR)
            if version_start == -1:
                continue
            model_class = AVAILABLE_MODELS[model]
            model_list.append((
                cp,
                model_class,
                f"hyps_{model}_{match.group()}"
            ))

    generate_hyps(
        model_list,
        sets,
        device,
        temps
    )


if __name__ == "__main__":
    CLI(main)
