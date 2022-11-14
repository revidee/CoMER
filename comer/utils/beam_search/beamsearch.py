import math
from typing import List, Union, Tuple

import torch
from torch import FloatTensor, LongTensor, Tensor
import torch.nn.functional as F
from comer.utils.beam_search import BeamManager, invalid_score
import itertools


class BatchedBeamSearch:
    """
        The BatchedBeamSearch class manages the complete variable-width BeamSearch for a single Batch.
        It supports bidirectional decoding and instantiates the needed BeamManagers for updating the active candidates.
        If no active beams have changed, the inputs from the batch will not be re-copied.
    """

    def __init__(self,
                 max_beams: int,
                 device: torch.device,
                 max_len: int,
                 bi_dir: bool = False,
                 find_top_k: int = 1,
                 max_candidates_per_node: int = 20,
                 absolute_pruning_threshold: float = float('Inf'),
                 relative_pruning_threshold: float = float('Inf'),
                 # relative_local_pruning_threshold: float = 5,
                 length_penalty: float = 1.0,
                 min_normalized_pseudo_probabilty: float = invalid_score,
                 temperature: float = 1.0,
                 debug: bool = False
                 ):
        self.max_beams = max_beams
        self.bi_dir = bi_dir
        self.find_top_k = find_top_k
        self.device = device
        self.max_len = max_len
        self.temperature = temperature
        self.empty_tensor = torch.empty((0,), device=device)

        self.length_penalty = length_penalty
        # Maximum Candidates Per Node
        self.mc = max_candidates_per_node
        # Absolute Threshold for Pruning
        self.ap_t = absolute_pruning_threshold
        self.min_normalized_hyp_score = min_normalized_pseudo_probabilty
        if min_normalized_pseudo_probabilty is not invalid_score:
            self.min_normalized_hyp_score = math.log(min_normalized_pseudo_probabilty)
        # Relative Threshold for Pruning
        self.rp_t = relative_pruning_threshold
        # Relative-Local Threshold for Pruning
        # self.rpl_t = relative_local_pruning_threshold
        # (2 - thresh) is needed since the score is a summed-log-prob and therefore negative.
        # But, if we want a condition, e.g.:
        #       score(candidate) > max_c{ score(c) } * rp (0.6=60%)
        #       (a candidate must be atleast 60% of the max candidate)
        # to hold, we have to make sure to transform 0.6 -> 1.4,
        #   s.t. a negative score is corrected downwards to respect the intended criteria.

        self.done = False
        self.beam_managers: List[BeamManager] = []
        self.original_src: Union[None, Tensor] = None
        self.original_src_mask: Union[None, Tensor] = None
        self.next_src: Union[None, Tensor] = None
        self.next_src_mask: Union[None, Tensor] = None
        self.next_input_ids: Union[None, Tensor] = None
        self.next_bm_refs: Union[None, Tensor] = None

        self.stats_repeated_src = 0
        self.stats_total_steps = 0

        self.debug = debug

    def reset(self):
        self.beam_managers = []
        self.done = False
        self.original_src = None
        self.original_src_mask = None
        self.next_src = None
        self.next_src_mask = None
        self.next_input_ids = None
        self.next_bm_refs = None
        self.stats_repeated_src = 0
        self.stats_total_steps = 0

    def predict(self,
                decode_step,
                src: FloatTensor,
                src_mask: LongTensor
                ):
        self.reset()
        batch_size = src.shape[0]
        # instantiate new beam managers.
        # Each Beam Manager analyzes the results from a single prediction step
        # and updates its state: creating new beams, pruning beams, being done, ...
        beam_managers: List[BeamManager] = [
            BeamManager(
                max_beams=self.max_beams,
                src_idx=src_idx,
                is_l2r=True,
                device=self.device,
                max_len=self.max_len,
                global_min_norm_threshold=self.min_normalized_hyp_score,
                save_best_k=self.find_top_k,
                debug=self.debug
            ) for src_idx in range(batch_size)
        ]

        if self.bi_dir:
            beam_managers = beam_managers + [
                BeamManager(
                    max_beams=self.max_beams,
                    src_idx=src_idx,
                    is_l2r=False,
                    device=self.device,
                    max_len=self.max_len,
                    global_min_norm_threshold=self.min_normalized_hyp_score,
                    save_best_k=self.find_top_k,
                    debug=self.debug
                ) for src_idx in range(batch_size)
            ]

        while True:
            self.generate_next_inputs(
                src, src_mask, beam_managers, self.device
            )
            if self.done:
                break

            topk_for_active_beams_scores, topk_for_active_beams_indices = torch.topk(
                # Get the top-k best entries for each active beam
                # k is limited by the beam_size, since we cannot have more beams in the following step
                #           TODO: Limit by maximum-candidates from variable-batch (+1 for EOS/SOS elimination),
                #            each beam can have at most mc following beams, except in the first iteration!
                F.log_softmax(
                    #   -> log-probabilities for all vocab-entries for each currently active beam
                    #   -> e.g. current active beams [[SOS, "1"]] (1 active beam)
                    #           result of F.log_softmax would then be of size [1, vocab-len]
                    #           to describe the log-likelihood of each "letter"
                    decode_step(self.next_src, self.next_src_mask, self.next_input_ids)[:, -1, :] / self.temperature,
                    dim=-1
                ),
                k=self.max_beams,
                sorted=True
            )
            current_beam_idx = 0
            for (bm_ref, amount_active_beams) in self.next_bm_refs:
                next_beam_idx = current_beam_idx + amount_active_beams
                beam_managers[bm_ref].update(topk_for_active_beams_scores[current_beam_idx:next_beam_idx, :],
                                             topk_for_active_beams_indices[current_beam_idx:next_beam_idx, :])
                current_beam_idx = next_beam_idx

        hyps_l2r: List[List[LongTensor]] = [[] for _ in itertools.repeat(None, batch_size)]
        scores_l2r: List[List[Tensor]] = [[] for _ in itertools.repeat(None, batch_size)]
        hyps_r2l = [[]]
        scores_r2l = [[]]
        if self.bi_dir:
            hyps_r2l: List[List[LongTensor]] = [[] for _ in itertools.repeat(None, batch_size)]
            scores_r2l: List[List[Tensor]] = [[] for _ in itertools.repeat(None, batch_size)]

        repeats = torch.zeros((src.shape[0],), dtype=torch.int, device=self.device)
        r2l_start_indices = torch.zeros((src.shape[0],), dtype=torch.int, device=self.device)
        for bm in beam_managers:
            best_hyps = bm.get_best_l2r_finalized()
            repeats[bm.src_idx] += len(best_hyps)
            if bm.is_direction_l2r:
                r2l_start_indices[bm.src_idx] += len(best_hyps)
                for (score, seq) in best_hyps:
                    hyps_l2r[bm.src_idx].append(seq)
                    scores_l2r[bm.src_idx].append(score.unsqueeze(0))
            else:
                for (score, seq) in best_hyps:
                    hyps_r2l[bm.src_idx].append(seq)
                    scores_r2l[bm.src_idx].append(score.unsqueeze(0))
        flattened_scores_l2r = list(itertools.chain.from_iterable(scores_l2r))
        flattened_scores_r2l = list(itertools.chain.from_iterable(scores_r2l))
        return list(itertools.chain.from_iterable(hyps_l2r)), \
               torch.cat(flattened_scores_l2r, dim=0) if len(flattened_scores_l2r) > 0 else self.empty_tensor, \
               list(itertools.chain.from_iterable(hyps_r2l)), \
               torch.cat(flattened_scores_r2l, dim=0) if len(flattened_scores_r2l) > 0 else self.empty_tensor, \
               repeats, \
               r2l_start_indices

    def generate_next_inputs(
            self,
            src: FloatTensor,
            src_mask: LongTensor,
            beam_managers: List[BeamManager],
            device: torch.device
    ):
        """Helper function to (re-)generate the correct model inputs based on the state of all beam managers.
            Parameters
            ----------
            src : FloatTensor
                [b, t, d]
            src_mask : LongTensor
                [b, t]
            beam_managers : List[BeamManager]
            device : torch.device       where to create the returned tensors
            """
        # How many times do we need to replicate each input from the batch?
        #   i.e. if the beam managers needs [0, 3] repeats, that means src[0] is repeated 0 times (not needed)
        #        and src[1] would be repeated 3 times, since there are 3 active beams.
        repeats = torch.zeros((src.shape[0],), dtype=torch.int, device=device)
        # Includes a list of all active beams, as list of lists, s.t. we can map beams to their src[i].
        current_ids: List[List[LongTensor]] = [[] for _ in itertools.repeat(None, src.shape[0])]
        # Also a list of lists, which contains back-references to beam managers
        # this is used to collect the correct portions of the decoding step and feed it to the beam managers to update
        # their step. E.g. [[(0,2), (3,2)], [(1,2)]] means bm[0] needs [0:2], bm[3] [2:4], bm[1] [5:6]
        # from the decoding output.
        bm_refs: List[List[Tuple[int, int]]] = [[] for _ in itertools.repeat(None, src.shape[0])]

        # if no beam manager has an active beam, there is nothing left to do.
        empty = True
        # If any beam manager has changed its amount of active beams, we need to "refresh" the src/src_mask before
        # we do another decoding step. Since BeamManagers can arbitrarily add/prune active beams, this might change at
        # each iteration. But if it doesn't we can re-use the existing one, without doing the same repeat_interleave.
        any_bm_beams_changed = False

        # loop micro-optimization
        current_ids_append = [l.append for l in current_ids]
        bm_refs_append = [l.append for l in bm_refs]

        for idx, bm in enumerate(beam_managers):
            # Did the BM change its beams?
            any_bm_beams_changed |= bm.beam_amount_changed_last_update
            bm.beam_amount_changed_last_update = False
            active_beams = bm.active_beams
            active_beams_size = bm.active_beams.shape[0]
            # If it has any active beams, record how many he needs for the next decoding step
            # With this, we can generate the next input src / src mask for the correct amount of beams
            # for each batched input
            if active_beams_size > 0:
                empty = False
                bm_src = bm.src_idx
                repeats[bm_src] += active_beams_size
                current_ids_append[bm_src](active_beams)
                bm_refs_append[bm_src]((idx, active_beams_size))

        if empty:
            self.done = True
            self.next_src = self.empty_tensor
            self.next_src_mask = self.empty_tensor
            self.next_input_ids = self.empty_tensor
            self.next_bm_refs = []
            return

        self.stats_total_steps += 1
        self.next_input_ids = torch.cat(list(itertools.chain.from_iterable(current_ids)), dim=0)
        if any_bm_beams_changed:
            self.stats_repeated_src += 1
            self.next_src = torch.repeat_interleave(src, repeats, dim=0)
            self.next_src_mask = torch.repeat_interleave(src_mask, repeats, dim=0)
            self.next_bm_refs = list(itertools.chain.from_iterable(bm_refs))
