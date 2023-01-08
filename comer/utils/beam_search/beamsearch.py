import math
from typing import List, Union

import torch
from torch import FloatTensor, LongTensor, Tensor
import torch.nn.functional as F

from comer.datamodules.crohme import vocab
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
                 max_candidates_per_node: int = 5,
                 absolute_pruning_threshold: float = 5,
                 relative_pruning_threshold: float = 2,
                 relative_pruning_offset: float = .45,
                 relative_local_pruning_threshold: float = 2,
                 relative_local_pruning_offset: float = .45,
                 length_penalty: float = 1.0,
                 min_normalized_pseudo_probabilty: float = invalid_score,
                 temperature: float = 1.0,
                 debug: bool = False,
                 save_logits: bool = False
                 ):
        self.max_beams = max_beams
        self.bi_dir = bi_dir
        self.find_top_k = find_top_k
        self.device = device
        self.max_len = max_len
        self.temperature = temperature
        self.save_logits = save_logits
        self.empty_tensor = torch.empty((0,), device=device)

        self.length_penalty = length_penalty

        self.min_normalized_hyp_score = min_normalized_pseudo_probabilty
        if min_normalized_pseudo_probabilty is not invalid_score:
            self.min_normalized_hyp_score = math.log(min_normalized_pseudo_probabilty)

        # Maximum Candidates Per Node
        self.mc = max_candidates_per_node
        # Absolute Threshold for Pruning
        self.ap_t = absolute_pruning_threshold
        # Relative Threshold for Pruning
        self.rp_t = relative_pruning_threshold
        self.rp_off = relative_pruning_offset
        # Relative-Local Threshold for Pruning
        self.rpl_t = relative_local_pruning_threshold
        self.rpl_off = relative_local_pruning_offset

        self.done = False
        self.beam_managers: List[BeamManager] = []
        self.next_bm_refs_rl_start_idx = 0
        self.original_src: Union[None, Tensor] = None
        self.original_src_mask: Union[None, Tensor] = None
        self.next_src: Union[None, Tensor] = None
        self.next_src_mask: Union[None, Tensor] = None
        self.next_input_ids: Union[None, Tensor] = None
        self.next_summed_logs_history: Union[None, Tensor] = None
        self.next_bm_refs: Union[None, List[int]] = None
        self.next_bm_active_beams: Union[None, LongTensor] = None
        self.next_invalid_token_mask: Union[None, Tensor] = None

        self.stats_repeated_src = 0
        self.stats_total_steps = 0

        self.sos_mask = torch.zeros((len(vocab)), dtype=torch.bool, device=self.device)
        self.sos_mask[vocab.SOS_IDX] = True
        self.eos_mask = torch.zeros((len(vocab)), dtype=torch.bool, device=self.device)
        self.eos_mask[vocab.EOS_IDX] = True

        self.debug = debug

    def reset(self):
        self.beam_managers = []
        self.done = False
        self.original_src = None
        self.original_src_mask = None
        self.next_src = None
        self.next_src_mask = None
        self.next_input_ids = None
        self.next_summed_logs_history = None
        self.next_summed_logs_history = None
        self.next_bm_refs = None
        self.next_bm_active_beams = None
        self.next_invalid_token_mask = None
        self.next_bm_refs_rl_start_idx = 0
        self.stats_repeated_src = 0
        self.stats_total_steps = 0

    def predict(self,
                decode_step,
                src: FloatTensor,
                src_mask: LongTensor
                ):
        self.reset()
        batch_size = src.shape[0]
        invalid_score_tensor = torch.tensor(invalid_score, device=self.device)
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
                save_best_k=self.find_top_k,
                debug=self.debug,
                length_penalty=self.length_penalty
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
                    save_best_k=self.find_top_k,
                    debug=self.debug,
                    length_penalty=self.length_penalty
                ) for src_idx in range(batch_size)
            ]
        self.beam_managers = beam_managers

        while True:
            self.generate_next_inputs(
                src, src_mask, beam_managers, self.device
            )
            if self.done:
                break
            unsharpened = decode_step(self.next_src, self.next_src_mask, self.next_input_ids)
            logits = F.log_softmax(
                #   -> log-probabilities for all vocab-entries for each active beam
                #   -> e.g. current active beams [[SOS, "1"]] (1 active beam)
                #           result of F.log_softmax would then be of size [1, vocab-len]
                #           to describe the log-likelihood of each "token"
                unsharpened[:, -1, :] / self.temperature,
                dim=-1
            )
            if not self.save_logits:
                unsharpened = None

            # Perform Invalid Sequence, Absolute, Relative, Relative Local Pruning, Maximum Candidates per Node
            #   c.f. https://arxiv.org/abs/1702.01806

            #####
            # Invalidate impossible sequences (i.e. l2r <sos>, 1, 2, <sos>):
            #####

            # Semantically, we want to do this:
            #   logits[0:self.next_bm_refs_rl_start_idx, vocab.SOS_IDX] = invalid_score_tensor
            #       (for all l2r beams, invalidate <SOS> tokens)
            #   logits[self.next_bm_refs_rl_start_idx:, vocab.EOS_IDX] = invalid_score_tensor
            #       (for all r2l beams, invalidate <EOS> tokens)
            # Which is, to set all <SOS> scores for L2R hypothesis to -Inf, s.t. they are never picked.

            # However, PyTorch indexing inefficiencies make this absurdly slow.
            #   Test on a P100 and 1 Batch (size 4) of the biggest images from CROHME 2014 Test Set: 46sec vs 36sec
            #   if using indexing vs. the following boolean mask approaches.
            # Thanks, PyTorch, I guess. Relevant GitHub Issue: https://github.com/pytorch/pytorch/issues/29973

            # Either use the boolean mask for indexing and setting it to an invalid score,
            # logits[self.next_invalid_token_mask] = invalid_score_tensor
            # or use the (slightly) faster variant of creating a new Tensor based on the boolean mask.
            logits = torch.where(self.next_invalid_token_mask, invalid_score, logits)

            #####
            # Maximum Candidates per Node
            #####

            # Get the top-"max-candidates-per-node" entries for each active beam.
            # Since we are limited to "max-candidates", we can never allow more from a single active beam, hence
            # we can simply drop any surplus.
            topk_wlogs_per_active_beam, topk_vocab_idx_per_active_beam = torch.topk(
                logits,
                k=self.mc,
                sorted=True
            )

            #####
            # Relative Local Threshold Pruning
            #####

            max_wlogs_per_active_beam = ((topk_wlogs_per_active_beam[:, 0]) - self.rpl_off) * self.rpl_t

            topk_wlogs_per_active_beam = torch.where(
                topk_wlogs_per_active_beam >= max_wlogs_per_active_beam.unsqueeze(-1).expand((-1, self.mc)),
                topk_wlogs_per_active_beam,
                invalid_score_tensor
            )

            # Calc the total summed logits for all possible next beams.
            topk_slogs_per_active_beam = topk_wlogs_per_active_beam

            if self.next_summed_logs_history.size(1) != 0:
                topk_slogs_per_active_beam += self.next_summed_logs_history[:, -1].unsqueeze(-1).expand(
                    (-1, self.mc)
                )

            #####
            # Global Threshold pruning
            #####
            normalized_logits = topk_slogs_per_active_beam / (
                    self.beam_managers[self.next_bm_refs[0]].curr_len * self.length_penalty
            )

            topk_slogs_per_active_beam = torch.where(
                normalized_logits >= self.min_normalized_hyp_score,
                topk_slogs_per_active_beam,
                invalid_score_tensor
            )

            #####
            # Absolute / Relative Threshold Pruning
            #####

            # Find the max. over all active candidates belonging to a single BeamManager.
            # But, each BeamManager might have a different amount of active beams.
            # For efficiency, we flatten all active beams (per BeamManager), to a single vector
            # and pad them to the same length.
            # The resulting big tensor can be used to calc the max per BeamManager in a single call.

            # references to views from the summed-logits tensor which contains all active beams for each BeamManager
            views_slogs = []
            v_slogs_append = views_slogs.append  # micro-optimization

            bm_offsets = [0]
            bm_offsets_append = bm_offsets.append

            current_beam_idx = 0
            for amount_active_beams in self.next_bm_active_beams:
                next_beam_idx = current_beam_idx + amount_active_beams
                bm_offsets_append(next_beam_idx)
                # v_wlogs_append(topk_wlogs_per_active_beams[current_beam_idx:next_beam_idx, :].view((-1)))
                v_slogs_append(topk_slogs_per_active_beam[current_beam_idx:next_beam_idx, :].view((-1)))
                current_beam_idx = next_beam_idx

            # Pad to equal length in a single tensor, in which each row corresponds to a single BeamManager
            flattened_padded_slogs_per_manager = torch.nn.utils.rnn.pad_sequence(views_slogs, batch_first=True,
                                                                                 padding_value=invalid_score)

            # Find the max
            max_per_manager, _ = torch.max(flattened_padded_slogs_per_manager, dim=1)

            # Mask Relative & Absolut Threshold pruning
            rel_thresh = (max_per_manager - self.rp_off) * self.rp_t
            abs_thresh = max_per_manager - self.ap_t
            max_tresh = torch.where(abs_thresh > rel_thresh, abs_thresh, rel_thresh)

            # Use a boolean mask to prune candidates that are not good enough
            filtered_per_bm = torch.where(
                flattened_padded_slogs_per_manager > max_tresh.unsqueeze(-1),
                flattened_padded_slogs_per_manager,
                invalid_score_tensor
            )

            min_total_beams_or_max_beams = min(filtered_per_bm.size(1), self.max_beams)
            next_beam_summed_logits, flattened_padded_per_bm_indices = torch.topk(
                filtered_per_bm,
                k=min_total_beams_or_max_beams,
                sorted=True
            )

            # transform from flattened indices to stacked indices
            # flattened layout: [<top-"max-cand" for beam 1> <top-"max-cand" for beam 2> ...]
            # gets the original beam index that got flattened.
            originating_active_beam_relative = torch.div(
                flattened_padded_per_bm_indices,
                self.mc,
                rounding_mode='trunc'
            )

            # BUT: if a lot gets pruned, a _padded_ -Inf might get transformed to a "fake" originating beam
            #   <top-mc 1> ... <top-mc n> <pad with -Inf to fit for parallel compute> <...>
            # So, limit each relative beam calculation to the known valid range.
            originating_active_beam_relative = torch.clamp(
                originating_active_beam_relative,
                max=(
                    (
                            torch.tensor(
                                self.next_bm_active_beams, device=self.device, dtype=torch.long
                            ) - 1
                    ).unsqueeze(-1))
            )

            # Add the beam manager offset to the relative offset to get the correct active beam index
            originating_active_beam_global = originating_active_beam_relative + torch.tensor(
                bm_offsets[:-1], device=self.device
            ).unsqueeze(-1).expand((-1, min_total_beams_or_max_beams))


            # use it to get & expand the sequence of the active beam with the best candidates.
            next_sequences = torch.cat(
                (
                    self.next_input_ids[originating_active_beam_global],
                    topk_vocab_idx_per_active_beam[
                        # row = the global active beam index
                        originating_active_beam_global,
                            # cols = what is the next token
                        flattened_padded_per_bm_indices % self.mc
                    ].unsqueeze(-1)
                ),
                dim=2
            )

            next_beam_summed_logits = next_beam_summed_logits.unsqueeze(-1)
            # print("----")
            if self.next_summed_logs_history.size(1) == 0:
                next_beam_summed_logits_history = next_beam_summed_logits
            else:
                next_beam_summed_logits_history = torch.cat(
                    (
                        self.next_summed_logs_history[originating_active_beam_global],
                        next_beam_summed_logits
                    ),
                    dim=-1
                )

                # print(self.next_summed_logs_history[originating_active_beam_global])
            # print(next_beam_summed_logits)
            # print("----")
            next_logits = None
            if self.save_logits:
                next_logits = unsharpened[originating_active_beam_global]

            # Update the cache of each beam manager, finish off hyps, check if done, gather the next active beams.
            for i, bm_ref in enumerate(self.next_bm_refs):
                beam_managers[bm_ref].update(
                    next_beam_summed_logits_history[i],
                    next_sequences[i],
                    raw_logits=None if not self.save_logits else next_logits[i]
                )

        # Now, each BeamManager has finished, either by reaching max_len or by having no active beams.
        # Gather all hypothesis per beam manager and save it at the correct location
        #   (corresponding src index and correct l2r / r2l split)
        hyps_l2r: List[List[LongTensor]] = [[] for _ in itertools.repeat(None, batch_size)]
        history_l2r: List[List[LongTensor]] = [[] for _ in itertools.repeat(None, batch_size)]
        scores_l2r: List[List[Tensor]] = [[] for _ in itertools.repeat(None, batch_size)]
        raw_logits_l2r: List[List[Tensor]] = [[] for _ in itertools.repeat(None, batch_size)]
        hyps_r2l: List[List[LongTensor]] = [[]]
        hyps_r2l_ori: List[List[LongTensor]] = [[]]
        history_r2l: List[List[LongTensor]] = [[]]
        scores_r2l: List[List[Tensor]] = [[]]
        raw_logits_r2l: List[List[Tensor]] = [[]]
        if self.bi_dir:
            hyps_r2l = [[] for _ in itertools.repeat(None, batch_size)]
            hyps_r2l_ori = [[] for _ in itertools.repeat(None, batch_size)]
            history_r2l = [[] for _ in itertools.repeat(None, batch_size)]
            scores_r2l = [[] for _ in itertools.repeat(None, batch_size)]
            raw_logits_r2l = [[] for _ in itertools.repeat(None, batch_size)]

        repeats_l2r = torch.zeros((src.shape[0],), dtype=torch.int, device=self.device)
        repeats_r2l = torch.zeros((src.shape[0],), dtype=torch.int, device=self.device)
        for bm in beam_managers:
            best_hyps = bm.get_best_l2r_finalized()
            if bm.is_direction_l2r:
                repeats_l2r[bm.src_idx] += len(best_hyps)
                for (score, seq, history, raw_logits, ori_seq) in best_hyps:
                    hyps_l2r[bm.src_idx].append(seq)
                    history_l2r[bm.src_idx].append(history)
                    scores_l2r[bm.src_idx].append(score.unsqueeze(0))
                    if self.save_logits:
                        raw_logits_l2r[bm.src_idx].append(raw_logits)
            else:
                repeats_r2l[bm.src_idx] += len(best_hyps)
                for (score, seq, history, raw_logits, ori_seq) in best_hyps:
                    hyps_r2l[bm.src_idx].append(seq)
                    hyps_r2l_ori[bm.src_idx].append(ori_seq)
                    history_r2l[bm.src_idx].append(history)
                    scores_r2l[bm.src_idx].append(score.unsqueeze(0))
                    if self.save_logits:
                        raw_logits_r2l[bm.src_idx].append(raw_logits)
        # flatten to a single list containing all hyps
        flattened_scores_l2r = list(itertools.chain.from_iterable(scores_l2r))
        flattened_scores_r2l = list(itertools.chain.from_iterable(scores_r2l))
        if self.save_logits:
            return list(itertools.chain.from_iterable(hyps_l2r)), \
                list(itertools.chain.from_iterable(history_l2r)), \
                torch.cat(flattened_scores_l2r, dim=0) if len(flattened_scores_l2r) > 0 else self.empty_tensor, \
                repeats_l2r, \
                list(itertools.chain.from_iterable(raw_logits_l2r)), \
                list(itertools.chain.from_iterable(hyps_r2l)), \
                list(itertools.chain.from_iterable(hyps_r2l_ori)), \
                list(itertools.chain.from_iterable(history_r2l)), \
                torch.cat(flattened_scores_r2l, dim=0) if len(flattened_scores_r2l) > 0 else self.empty_tensor, \
                repeats_r2l, \
                list(itertools.chain.from_iterable(raw_logits_r2l))
        return list(itertools.chain.from_iterable(hyps_l2r)), \
            list(itertools.chain.from_iterable(history_l2r)), \
            torch.cat(flattened_scores_l2r, dim=0) if len(flattened_scores_l2r) > 0 else self.empty_tensor, \
            repeats_l2r, \
            list(itertools.chain.from_iterable(hyps_r2l)), \
            list(itertools.chain.from_iterable(hyps_r2l_ori)), \
            list(itertools.chain.from_iterable(history_r2l)), \
            torch.cat(flattened_scores_r2l, dim=0) if len(flattened_scores_r2l) > 0 else self.empty_tensor, \
            repeats_r2l

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
        repeats_l2r = torch.zeros((src.shape[0],), dtype=torch.int, device=device)
        repeats_r2l = torch.zeros((src.shape[0],), dtype=torch.int, device=device)

        # The following list of lists hold data for each active beam.
        # We collect the beams from each beam managers, which can arbitrarily reference to an input/src index.
        # Hence, we have to append them to different lists, corresponding to each src index.
        # Later, we flatten the lists of lists to tensors / or a single list

        # Each BeamManager holds a Tensor of its active beams and their input sequences.
        # Here, we save them to the correct src-index-list
        current_ids_l2r: List[List[LongTensor]] = [[] for _ in itertools.repeat(None, src.shape[0])]
        current_ids_r2l: List[List[LongTensor]] = [[] for _ in itertools.repeat(None, src.shape[0])]
        # Additionally, each BeamManagers holds a tensor of the summed logits for each active beam.
        current_sum_logits_l2r: List[List[LongTensor]] = [[] for _ in itertools.repeat(None, src.shape[0])]
        current_sum_logits_r2l: List[List[LongTensor]] = [[] for _ in itertools.repeat(None, src.shape[0])]
        # In order to correlate an active beam back to the managing BeamManager, we save
        # 1) the order of beam managers
        bm_refs_l2r: List[List[int]] = [[] for _ in itertools.repeat(None, src.shape[0])]
        bm_refs_r2l: List[List[int]] = [[] for _ in itertools.repeat(None, src.shape[0])]
        # 2) the amount of active beams for each beam manager
        bm_active_beams_l2r: List[List[int]] = [[] for _ in itertools.repeat(None, src.shape[0])]
        bm_active_beams_r2l: List[List[int]] = [[] for _ in itertools.repeat(None, src.shape[0])]

        # if no beam manager has an active beam, there is nothing left to do.
        empty_l2r = True
        empty_r2l = True
        # If any beam manager has changed its amount of active beams, we need to "refresh" the src/src_mask before
        # we do another decoding step. Since BeamManagers can arbitrarily add/prune active beams, this might change at
        # each iteration. But if it doesn't we can re-use the existing one, without doing the same repeat_interleave.
        any_bm_beams_changed = False

        active_beams_l2r = 0
        active_beams_r2l = 0

        active_bms_l2r = 0
        current_bm_offset = 0

        for idx, bm in enumerate(beam_managers):
            # Did the BM change its beams, if so, we need to recalc the input tensors for inference
            any_bm_beams_changed |= bm.beam_amount_changed_last_update
            bm.beam_amount_changed_last_update = False

            active_beams = bm.active_beams
            sum_logits = bm.active_beams_summed_logits
            active_beams_size = bm.active_beams.shape[0]
            # If it has any active beams, record how many he needs for the next decoding step
            # With this, we can generate the next input src / src mask for the correct amount of beams
            # for each batched input
            if active_beams_size > 0:
                bm_src = bm.src_idx
                if bm.is_direction_l2r:
                    empty_l2r = False
                    repeats_l2r[bm_src] += active_beams_size
                    active_beams_l2r += active_beams_size
                    current_ids_l2r[bm_src].append(active_beams)
                    current_sum_logits_l2r[bm_src].append(sum_logits)
                    bm_refs_l2r[bm_src].append(idx)
                    bm_active_beams_l2r[bm_src].append(active_beams_size)
                    active_bms_l2r += 1
                else:
                    empty_r2l = False
                    repeats_r2l[bm_src] += active_beams_size
                    active_beams_r2l += active_beams_size
                    current_ids_r2l[bm_src].append(active_beams)
                    current_sum_logits_r2l[bm_src].append(sum_logits)
                    bm_refs_r2l[bm_src].append(idx)
                    bm_active_beams_r2l[bm_src].append(active_beams_size)
                current_bm_offset += active_beams_size

        if empty_l2r and empty_r2l:
            self.done = True
            self.next_src = self.empty_tensor
            self.next_src_mask = self.empty_tensor
            self.next_input_ids = self.empty_tensor
            self.next_bm_refs = []
            return

        self.stats_total_steps += 1
        self.next_input_ids = torch.cat(
            list(
                itertools.chain(
                    itertools.chain.from_iterable(current_ids_l2r),
                    itertools.chain.from_iterable(current_ids_r2l)
                )
            ),
            dim=0
        )
        self.next_summed_logs_history = torch.cat(
            list(
                itertools.chain(
                    itertools.chain.from_iterable(current_sum_logits_l2r),
                    itertools.chain.from_iterable(current_sum_logits_r2l)
                )
            ),
            dim=0
        )
        if any_bm_beams_changed:
            # we have to refresh next_src / src_mask / invalidation masks
            # Since the amount of active beams has changed
            self.stats_repeated_src += 1

            self.next_bm_refs_rl_start_idx = active_bms_l2r
            self.next_bm_refs = list(itertools.chain(
                itertools.chain.from_iterable(bm_refs_l2r), itertools.chain.from_iterable(bm_refs_r2l)
            ))

            self.next_bm_active_beams = list(itertools.chain(
                itertools.chain.from_iterable(bm_active_beams_l2r), itertools.chain.from_iterable(bm_active_beams_r2l)
            )
            )

            if not empty_l2r:
                self.next_src = torch.repeat_interleave(src, repeats_l2r, dim=0)
                self.next_src_mask = torch.repeat_interleave(src_mask, repeats_l2r, dim=0)
                self.next_invalid_token_mask = self.sos_mask.expand((active_beams_l2r, -1))
                if not empty_r2l:
                    self.next_src = torch.cat(
                        (self.next_src, torch.repeat_interleave(src, repeats_r2l, dim=0)), dim=0
                    )
                    self.next_src_mask = torch.cat(
                        (self.next_src_mask, torch.repeat_interleave(src_mask, repeats_r2l, dim=0)), dim=0
                    )
                    self.next_invalid_token_mask = torch.cat(
                        (self.next_invalid_token_mask, self.eos_mask.expand(
                            (active_beams_r2l, -1))
                         ), dim=0
                    )
            else:
                self.next_src = torch.repeat_interleave(src, repeats_r2l, dim=0)
                self.next_src_mask = torch.repeat_interleave(src_mask, repeats_r2l, dim=0)
                self.next_invalid_token_mask = self.eos_mask.expand(
                    (active_beams_r2l, -1)
                )
