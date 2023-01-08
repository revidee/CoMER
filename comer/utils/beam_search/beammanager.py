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
                 debug: bool = False,
                 length_penalty: float = 1.0
                 ):
        # static / rather static variables, calculated/set once for easy access
        self.max_beams = max_beams
        self.src_idx = src_idx
        self.is_direction_l2r = is_l2r
        self.device = device
        self.max_len = max_len
        self.save_best_k = save_best_k
        self.length_penalty = length_penalty

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
        self.active_beams_summed_logits = torch.empty((0, 0), dtype=torch.long, device=self.device)

        # List of finished hypothesis as tuple (score, sequence, summed logit history)
        self.best_hyps: List[Tuple[Tensor, Tensor, FloatTensor, Union[FloatTensor, None]]] = []
        self.worst_score: Tensor = invalid_score
        self.done = False
        self.debug = debug

    def __str__(self):
        return f"beam_manager[{self.src_idx}]({'l2r' if self.is_direction_l2r else 'r2l'})"

    def update(self, summed_logits: FloatTensor, sequences: LongTensor, raw_logits: FloatTensor = None):
        """
            Iterates over the summed logits and sequences of all possible & best candidates.
            However, it stops as soon as it encounters an "-Inf" score which indicates a pruned candidate.
            Since summed_logits is sorted, we can safely abort early without iterating over all candidates.

            It then checks for finished candidates and saves them into a best-hypothesis cache, if the finished
            candidate is better than the current worst one cached.

            Finally, it checks if there are any new active beams or if any active beam can get better than the current
            worst one. If not, we can abort early.

            Parameters
            ----------
            summed_logits : FloatTensor
                [max_beams, self.curr_len + 1]
                A tensor containing logit history and the logit of the currently predicted token.
                Limited to the maximum amount of beams.
            sequences : LongTensor
                [max_beams, self.curr_len + 1]
                A tensor containing the complete sequences for the next iteration. Therefore, it contains one more
                entry than the current length.
        """
        curr_normalizing_fac = (self.curr_len ** self.length_penalty)

        next_active_beam_indices = []
        for i, logit_history in enumerate(summed_logits):
            current_token_logit = logit_history[-1]
            # If we encounter an -inf in the _sorted_ candidates,
            # we can safely abort since we then reached the invalidated ones.
            if current_token_logit == self.invalid_score_tensor:
                break
            # get the actual token and construct the full sequence
            if sequences[i][sequences.size(1) - 1] == self.end_token_tensor:
                if self.debug:
                    print("fin", vocab.indices2words(sequences[i].tolist()), current_token_logit / curr_normalizing_fac)
                # if the beam has ended, cache it as best or drop it immediately
                # self.finish_single(normalized_log_probs[originating_beam_idx][originating_beam_topk_idx], next_sequence)
                self.finish_single(current_token_logit / curr_normalizing_fac, sequences[i], logit_history, raw_logits[i] if raw_logits is not None else None)
            else:
                if self.debug:
                    print("add beam", vocab.indices2words(sequences[i].tolist()), current_token_logit)
                # Add sequence with it's summed log-probability to the next active beams
                next_active_beam_indices.append(i)
        if self.debug:
            print("---")
            print("")

        next_active_beams_len = len(next_active_beam_indices)
        self.beam_amount_changed_last_update = self.active_beams_len != next_active_beams_len
        self.active_beams_len = next_active_beams_len

        if next_active_beams_len == 0 or self.curr_len == self.max_len or self.worst_score >= (
                summed_logits[next_active_beam_indices[0], -1] / curr_normalizing_fac
        ):
            self.done = True
            self.active_beams = torch.empty((0,), device=self.device)
            self.active_beams_summed_logits = torch.empty((0,), device=self.device)
            self.active_beams_len = 0
            self.beam_amount_changed_last_update = True
            return
        self.active_beams = sequences[next_active_beam_indices]
        self.active_beams_summed_logits = summed_logits[next_active_beam_indices]

        self.curr_len = self.curr_len + 1

    def finish_single(self, score: Tensor, sequence: Tensor, history: FloatTensor, raw_logits: FloatTensor = None):
        best_hyp_len = len(self.best_hyps)
        if best_hyp_len == 0:
            self.best_hyps.append((score, sequence, history, raw_logits))
            self.worst_score = score
            return
        # find place to insert
        last_idx_better_than = 0
        for i, (best_hyp_score, _, _, _) in enumerate(self.best_hyps):
            if best_hyp_score >= score:
                last_idx_better_than = i
            else:
                break
        # first one that is potentially worse
        last_idx_better_than += 1

        if best_hyp_len < self.save_best_k:
            self.best_hyps.insert(last_idx_better_than, (score, sequence, history, raw_logits))
            self.worst_score = self.best_hyps[-1][0]
            return
        # check if we have to replace it
        if last_idx_better_than != best_hyp_len and score >= self.best_hyps[last_idx_better_than][0]:
            self.best_hyps[last_idx_better_than] = (score, sequence, history, raw_logits)
            self.worst_score = self.best_hyps[-1][0]

    def get_best_l2r_finalized(self) -> Union[None, List[Tuple[Tensor, Tensor, Tensor, Union[Tensor, None], Tensor]]]:
        """
            Returns the best hypothesis found and removes the SOS and EOS token
        """
        if len(self.best_hyps) == 0:
            return self.best_hyps
        output = []
        for (score, seq, history, raw_logs) in self.best_hyps:
            summed_history = history[:-1]
            shifted_history = torch.zeros((summed_history.size(0)), device=self.device)
            shifted_history[1:] = summed_history[:-1]
            single_logits = summed_history - shifted_history
            if self.is_direction_l2r:
                output.append((score, seq[1:-1], single_logits, raw_logs, seq[1:-1]))
            else:
                output.append((score, torch.flip(seq[1:-1], dims=[0]), torch.flip(single_logits, dims=[0]), raw_logs, seq[1:-1]))
        return output
