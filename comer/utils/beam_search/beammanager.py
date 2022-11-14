from typing import Union, Tuple, List

import torch
from torch import Tensor, FloatTensor, LongTensor

from comer.datamodules.crohme import vocab
from comer.utils.beam_search import invalid_score


class BeamManager:
    """
        A BeamManager is responsible for performing a variable-width beam search for a single input.
        It receives the (top-k) decoding outputs and prunes the possible candidate choices.
        When done (i.e. no active beams left), the best hypothesis can be retrieved.
    """
    def __init__(self,
                 max_beams: int,
                 src_idx: int,
                 is_l2r: bool,
                 device: torch.device,
                 max_len: int,
                 save_best_k: int = 1,
                 max_candidates_per_node: int = 10,
                 absolute_pruning_threshold: float = 5,
                 relative_pruning_threshold: float = 2,
                 relative_pruning_offset: float = .45,
                 relative_local_pruning_threshold: float = 2,
                 relative_local_pruning_offset: float = .45,
                 length_penalty: float = 1.0,
                 global_min_norm_threshold: float = invalid_score,
                 debug: bool = False
                 ):
        # static / rather static variables, calculated/set once for easy access
        self.max_beams = max_beams
        self.src_idx = src_idx
        self.is_direction_l2r = is_l2r
        self.device = device
        self.max_len = max_len
        self.save_best_k = save_best_k

        self.length_penalty = length_penalty
        # Maximum Candidates Per Node
        self.mc = max_candidates_per_node
        # Absolute Threshold for Pruning
        self.ap_t = absolute_pruning_threshold
        self.min_normalized_hyp_score = global_min_norm_threshold
        # Relative Threshold for Pruning
        self.rp_t = relative_pruning_threshold
        self.rp_off = relative_pruning_offset
        # Relative-Local Threshold for Pruning
        self.rpl_t = relative_local_pruning_threshold
        self.rpl_off = relative_local_pruning_offset

        # constant "-inf" tensor to filter out invalid/pruned beams
        self.invalid_score_tensor = torch.tensor(float('-Inf'), device=device)
        # start token, depending on the decoding direction
        self.start_token = vocab.SOS_IDX if is_l2r else vocab.EOS_IDX
        # end token, depending on the decoding direction
        self.end_token = vocab.EOS_IDX if is_l2r else vocab.SOS_IDX
        # end token as tensor, for easy comparison without casting
        self.end_token_tensor \
            = torch.tensor(vocab.EOS_IDX, device=device) if is_l2r else torch.tensor(vocab.SOS_IDX, device=device)

        # state variables
        self.curr_len = 1
        self.beam_amount_changed_last_update = True

        self.active_beams = torch.full(
            (1, 1),
            fill_value=self.start_token,
            dtype=torch.long,
            device=self.device,
        )
        self.active_beams_len = 1
        self.active_beams_summed_logits = torch.zeros(1, dtype=torch.long, device=self.device)

        # List of finished hypothesis as tuple (score, sequence)
        self.best_hyps: List[Tuple[Tensor, Tensor]] = []
        self.done = False
        self.debug = debug

    def __str__(self):
        return f"beam_manager[{self.src_idx}]({'l2r' if self.is_direction_l2r else 'r2l'})"

    def update(self, word_log_probs: FloatTensor, vocab_indices: LongTensor):
        # prune invalid next tokens (i.e. we started with SOS, then generating a SOS token is forbidden and is pruned)

        word_log_probs = torch.where(vocab_indices == self.start_token, invalid_score, word_log_probs)

        # add the summed score to it, to compute the final scores for the active candidates of equal length
        #   we don't need to normalize this by the sequence length, because of equality of their length
        # Only if we want to add a finished beam to the output collection
        #   we may normalize it by the length and add a length penalty.

        # Perform Absolute, Relative, Relative Local Pruning + Global Threshold Pruning
        #   c.f. https://arxiv.org/abs/1702.01806

        # compute the single best next token, i.e. max(score_word(candidate)) over all candidates
        #   This is used for "Relative Local Threshold Pruning"
        if self.debug:
            print(f"update {self}")
            word_logs_list = word_log_probs.tolist()
            for i, wl in enumerate(vocab_indices.tolist()):
                print(vocab.indices2words(self.active_beams[i].tolist()))
                for wi, w in enumerate(wl):
                    print(f"[{vocab.idx2word[w]} {word_logs_list[i][wi]:.2f}] ", end="")
                print()
        # flattened_word_log_probs = word_log_probs.view(-1)
        # max_cand_wscore_idx = torch.argmax(flattened_word_log_probs)
        # max_cand_wscore = flattened_word_log_probs[max_cand_wscore_idx]

        # max_cand_wscore_old, _ = torch.max(word_log_probs, dim=1)
        max_per_beam = torch.index_select(word_log_probs, dim=1,
                                             index=(vocab_indices[:, 0] == self.start_token).long())[:, 0]
        # print("wlogs: ", word_log_probs)
        # print("old: ", max_cand_wscore_old)
        # print("new: ", max_cand_wscore)
        max_cand_wscore = (
                (
                        torch.where(
                            abs(max_per_beam) < 0.0001, invalid_score, max_per_beam
                        ) - self.rpl_off
                ) * self.rpl_t
        ).unsqueeze(-1)
        # print("new after: ", max_cand_wscore)

        # compute the single best active candidate, i.e. max(score(candidate)) over all candidates
        #   This is used for both "Absolute" & "Relative Threshold Pruning"
        summed_log_probs = word_log_probs + self.active_beams_summed_logits.unsqueeze(-1).expand(word_log_probs.size())
        max_cand_score = torch.max(max_per_beam + self.active_beams_summed_logits)

        if self.debug:
            print(f"max wscore", max_cand_wscore)
            print(f"max score", max_cand_score)

        # compute the normalized score, s.t. scores of different lengths are comparable
        curr_normalizing_fac = (self.curr_len ** self.length_penalty)
        normalized_log_probs = summed_log_probs / curr_normalizing_fac
        if self.debug:
            print("summed_log_probs")
            print(summed_log_probs)
            print("normalized_log_probs")
            print(normalized_log_probs)

        # Create a mask which excludes all pruned candidates
        rel_thresh = (max_cand_score - self.rp_off) * self.rp_t
        # rel_local_thresh = (max_cand_wscore - self.rpl_off) * self.rpl_t
        abs_thresh = max_cand_score - self.ap_t

        rel_thresh = invalid_score if rel_thresh == 0 else rel_thresh
        # rel_local_thresh = invalid_score if rel_local_thresh == 0 else rel_local_thresh
        abs_thresh = invalid_score if abs_thresh == 0 else abs_thresh

        upper_rel_abs_thresh = max((rel_thresh, abs_thresh))

        pass_mask = (
                # Only candidates with a larger normalized score are considered (Global Pruning)
                # (normalized_log_probs >= self.min_normalized_hyp_score)
                # A full-score must be in X% range & X-range of the single best score
                (summed_log_probs >= upper_rel_abs_thresh)
                # A word-score must be in X% range of the single best word
                & (word_log_probs >= max_cand_wscore)
        )
        # Do the actual masking, now valid_summed_log_probs either contains a summed score, or -inf
        valid_summed_log_probs = torch.where(pass_mask, summed_log_probs, invalid_score)
        if self.debug:
            print(f"rel: {rel_thresh}, abs: {abs_thresh}")
            print("valids")
            print(valid_summed_log_probs)
        # Without Maximum Candidates, we could simply now choose the best k next candidates as active beams
        # and continue.
        # But in order to limit the max. next beams per current active beam, we need to first
        # calculate the top-(max-candidates) for each active beam
        max_candidates = (
            min(self.max_beams, valid_summed_log_probs.size(1)) if self.curr_len == 1 else min(self.max_beams, self.mc)
        )
        top_mc_per_beam, top_mc_per_beam_indices = torch.topk(valid_summed_log_probs,
                                                              max_candidates,
                                                              sorted=False)
        top_mc_per_beam_indices_flattened = top_mc_per_beam_indices.view(-1)
        top_mc_per_beam_flattened = top_mc_per_beam.view(-1)

        # and then choose the top-(beam-size) candidates as the next ones
        # This way, we can guarantee, that we have at most (max-candidates) as direct descendants from each active beam.
        top_cand_scores, top_cand_topk_indices = torch.topk(top_mc_per_beam_flattened,
                                                            min(top_mc_per_beam_flattened.size(0), self.max_beams))
        next_active_beams = []
        next_active_beams_summed_logits = []
        if self.debug:
            print("top cand scores:")
            print(top_cand_scores)
        for i, val in enumerate(top_cand_scores):
            # If we encounter an -inf in the _sorted_ candidates,
            # we can safely abort since we then reached the invalidated ones.
            if val == self.invalid_score_tensor:
                break
            # now select the according best next token and add it to the sequence it originated from
            originating_beam_idx = torch.div(top_cand_topk_indices[i], max_candidates, rounding_mode="floor")
            originating_beam_topk_idx = top_mc_per_beam_indices_flattened[top_cand_topk_indices[i]]
            # get the actual token and construct the full sequence
            next_token = vocab_indices[originating_beam_idx][originating_beam_topk_idx]
            next_sequence = torch.cat((
                self.active_beams[originating_beam_idx, :],
                next_token.unsqueeze(-1)
            ))

            if next_token == self.end_token:
                if self.debug:
                    print("fin", vocab.indices2words(next_sequence.tolist()), self.active_beams_summed_logits[originating_beam_idx] / curr_normalizing_fac)
                # if the beam has ended, cache it as best or drop it immediately
                # self.finish_single(normalized_log_probs[originating_beam_idx][originating_beam_topk_idx], next_sequence)
                self.finish_single(self.active_beams_summed_logits[originating_beam_idx] / curr_normalizing_fac, next_sequence)
            else:
                if self.debug:
                    print("add beam", vocab.indices2words(next_sequence.tolist()), summed_log_probs[originating_beam_idx][originating_beam_topk_idx])
                # Add sequence with it's summed log-probability to the next active beams
                next_active_beams.append(next_sequence.unsqueeze(0))
                next_active_beams_summed_logits.append(
                    (summed_log_probs[originating_beam_idx][originating_beam_topk_idx]).unsqueeze(-1)
                )
        if self.debug:
            print("---")
            print("")
        next_active_beams_len = len(next_active_beams)
        self.beam_amount_changed_last_update = self.active_beams_len != next_active_beams_len
        self.active_beams_len = next_active_beams_len
        if next_active_beams_len == 0 or self.curr_len == self.max_len:
            self.done = True
            self.active_beams = torch.empty((0,), device=self.device)
            self.active_beams_summed_logits = torch.empty((0,), device=self.device)
            return
        self.active_beams = torch.cat(next_active_beams)
        self.active_beams_summed_logits = torch.cat(next_active_beams_summed_logits)

        self.curr_len = self.curr_len + 1

    def finish_single(self, score: Tensor, sequence: Tensor):
        if len(self.best_hyps) < self.save_best_k:
            self.best_hyps.append((score, sequence))
            return
        # select the worst one to replace
        curr_worst_idx = -1,
        curr_worst = float('Inf')
        for i, (best_hyp_score, _) in enumerate(self.best_hyps):
            if best_hyp_score < curr_worst:
                curr_worst = best_hyp_score
                curr_worst_idx = i
        if curr_worst_idx != -1 and score > curr_worst:
            self.best_hyps[curr_worst_idx] = (score, sequence)

    def get_best_l2r_finalized(self) -> Union[None, List[Tuple[Tensor, Tensor]]]:
        """
            Returns the best hypothesis found and removes the SOS and EOS token
        """
        if len(self.best_hyps) == 0:
            return self.best_hyps
        if self.is_direction_l2r:
            return [(score, seq[1:-1]) for (score, seq) in self.best_hyps]
        return [(score, torch.flip(seq[1:-1], dims=[0])) for (score, seq) in self.best_hyps]