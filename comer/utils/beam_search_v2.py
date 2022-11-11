import math
from typing import Tuple, Union

import torch
from comer.datamodules.crohme import vocab
from torch import FloatTensor, LongTensor, Tensor


invalid_score = float('-Inf')


class BeamManager:
    def __init__(self,
                 max_beams: int,
                 src_idx: int,
                 is_l2r: bool,
                 device: torch.device,
                 max_len: int,
                 max_candidates_per_node: int = 3,
                 absolute_pruning_threshold: float = 2.5,
                 relative_pruning_threshold: float = 0.6,
                 relative_local_pruning_threshold: float = 0.02,
                 length_penalty: float = 1.0,
                 min_normalized_pseudo_probabilty: float = invalid_score
                 ):
        # static / rather static variables, calculated/set once for easy access
        self.max_beams = max_beams
        self.src_idx = src_idx
        self.is_direction_l2r = is_l2r
        self.device = device
        self.max_len = max_len

        self.length_penalty = length_penalty
        # Maximum Candidates Per Node
        self.mc = max_candidates_per_node
        # Absolute Threshold for Pruning
        self.ap_t = absolute_pruning_threshold
        self.min_normalized_hyp_score = min_normalized_pseudo_probabilty
        if min_normalized_pseudo_probabilty is not invalid_score:
            self.min_normalized_hyp_score = math.log(min_normalized_pseudo_probabilty)
        # Relative Threshold for Pruning
        self.rp_t = (2 - relative_pruning_threshold)
        # Relative-Local Threshold for Pruning
        self.rpl_t = (2 - relative_local_pruning_threshold)
        # (2 - thresh) is needed since the score is a summed-log-prob and therefore negative.
        # But, if we want a condition, e.g.:
        #       score(candidate) > max_c{ score(c) } * rp (0.6=60%)
        #       (a candidate must be atleast 60% of the max candidate)
        # to hold, we have to make sure to transform 0.6 -> 1.4,
        #   s.t. a negative score is corrected downwards to respect the intended criteria.

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
        self.current_beam_amount_changed = False

        self.active_beams = torch.full(
            (1, 1),
            fill_value=self.start_token,
            dtype=torch.long,
            device=self.device,
        )
        self.active_beams_summed_logits = torch.zeros(1, dtype=torch.long, device=self.device)

        # List of finished hypothesis as tuple (score, sequence)
        self.best_hyp: Union[None, Tuple[Tensor, Tensor]] = None
        self.done = False

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
        flattened_word_log_probs = word_log_probs.view(-1)
        max_cand_wscore_idx = torch.argmax(flattened_word_log_probs)
        max_cand_wscore = flattened_word_log_probs[max_cand_wscore_idx]

        # compute the single best active candidate, i.e. max(score(candidate)) over all candidates
        #   This is used for both "Absolute" & "Relative Threshold Pruning"
        summed_log_probs = word_log_probs + self.active_beams_summed_logits.unsqueeze(-1).expand(word_log_probs.size())
        flattened_summed_logs_probs = summed_log_probs.view(-1)
        max_cand_score_idx = torch.argmax(flattened_summed_logs_probs)
        max_cand_score = flattened_summed_logs_probs[max_cand_score_idx]

        # compute the normalized score, s.t. scores of different lengths are comparable
        curr_normalizing_fac = (self.curr_len ** self.length_penalty)
        normalized_log_probs = summed_log_probs / curr_normalizing_fac

        # Create a mask which excludes all pruned candidates
        rel_thresh = max_cand_score * self.rp_t
        rel_local_thresh = max_cand_wscore * self.rpl_t
        abs_thresh = max_cand_score - self.ap_t

        pass_mask = (
                # Only candidates with a larger normalized score are considered (Global Pruning)
                (normalized_log_probs >= self.min_normalized_hyp_score)
                # A full-score must be in X% range of the single best score
                & (summed_log_probs >= rel_thresh)
                # A full-score must be in range-X of the single best score (i.e. closer than 2.5 in log-prob space)
                & (summed_log_probs >= abs_thresh)
                # A word-score must be in X% range of the single best word
                & (word_log_probs >= rel_local_thresh)
        )
        # Do the actual masking, now valid_summed_log_probs either contains a summed score, or -inf
        valid_summed_log_probs = torch.where(pass_mask, summed_log_probs, invalid_score)

        # Without Maximum Candidates, we could simply now choose the best k next candidates as active beams
        # and continue.
        # But in order to limit the max. next beams per current active beam, we need to first
        # calculate the top-(max-candidates) for each active beam
        max_candidates = min(self.max_beams, valid_summed_log_probs.size(1)) if self.curr_len == 1 else self.mc
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
                # if the beam has ended, cache it as best or drop it immediately
                self.finish_single(normalized_log_probs[originating_beam_idx][originating_beam_topk_idx], next_sequence)
            else:
                # Add sequence with it's summed log-probability to the next active beams
                next_active_beams.append(next_sequence.unsqueeze(0))
                next_active_beams_summed_logits.append(
                    (summed_log_probs[originating_beam_idx][originating_beam_topk_idx]).unsqueeze(-1)
                )
        if len(next_active_beams) == 0 or self.curr_len == self.max_len:
            self.done = True
            self.active_beams = torch.empty((0,), device=self.device)
            self.active_beams_summed_logits = torch.empty((0,), device=self.device)
            return

        self.active_beams = torch.cat(next_active_beams)
        self.active_beams_summed_logits = torch.cat(next_active_beams_summed_logits)

        self.curr_len = self.curr_len + 1

    def finish_single(self, score: Tensor, sequence: Tensor):
        if self.best_hyp is None:
            self.best_hyp = (score, sequence)
        else:
            if score > self.best_hyp[0]:
                self.best_hyp = (score, sequence)

    def get_best_l2r_finalized(self) -> Union[None, Tuple[Tensor, Tensor]]:
        """
            Returns the best hypothesis found and removes the SOS and EOS token
        """
        if self.best_hyp is None:
            return self.invalid_score_tensor, torch.empty(0, device=self.device)
        if self.is_direction_l2r:
            return self.best_hyp[0], self.best_hyp[1][1:-1]

        return self.best_hyp[0], torch.flip(self.best_hyp[1], dims=[0])[1:-1]

