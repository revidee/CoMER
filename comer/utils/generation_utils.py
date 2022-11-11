from abc import abstractmethod
import itertools as iter
from typing import List, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from comer.datamodules.crohme import vocab, vocab_size
from comer.utils.utils import Hypothesis, ce_loss, to_tgt_output
from einops import rearrange
from einops.einops import repeat
from torch import FloatTensor, LongTensor, Tensor

from .beam_search import BeamSearchScorer
from .beam_search_v2 import BeamManager


# modified from
# https://github.com/huggingface/transformers/blob/af6e01c5bc39467f1e3ce47a2135fb1777af1db2/src/transformers/generation_utils.py#L1843


class DecodeModel(pl.LightningModule):
    @abstractmethod
    def transform(
            self, src: List[FloatTensor], src_mask: List[LongTensor], input_ids: LongTensor
    ) -> FloatTensor:
        """decode one step

        Parameters
        ----------
        src : List[FloatTensor]
            [b, t, d]
        src_mask : List[LongTensor]
            [b, t]
        input_ids : LongTensor
            [b, l]

        Returns
        -------
        FloatTensor
            [b, l, vocab_size]
        """
        raise NotImplementedError("This is an abstract method.")

    def beam_search(
            self,
            src: [FloatTensor],
            src_mask: [LongTensor],
            beam_size: int,
            max_len: int,
            alpha: float,
            early_stopping: bool,
            temperature: float,
            bi_dir: bool = True,
            scoring_run: bool = True,
    ) -> List[Hypothesis]:
        """run beam search to decode

        Parameters
        ----------
        src : List[FloatTensor]
            [b, t, d]
            List, because _beam_search function mutates the tensor directly
            It could return the mutated new_src however...
        src_mask : List[LongTensor]
            [b, t]
            List, because _beam_search function mutates the tensor directly
            It could return the mutated new_src_mask however...
        beam_size : int
        max_len : int
        alpha : float
        early_stopping : bool

        Returns
        -------
        List[Hypothesis]: [batch_size,]
        """
        # If we want to run bi-directional beam_search, we generate double the hypothesis.
        # We aim to generate k-best Hyps for each - L2R and R2L - decoding.
        # Thus, we have to double the batch, s.t. we start with
        # (batch * 2 * beam_size) Beams which we explore to generate (batch * 2 * beam_size) Hyps
        single_direction_size = src[0].shape[0]
        if bi_dir:
            batch_size = single_direction_size * 2
            # original batch size (which we double) times beam_size = start idx of the r2l decoding
            r2l_start_idx = single_direction_size * beam_size

            # [2 * b, t, d], [l2r l2r, r2l r2l]
            src[0] = torch.cat((src[0], src[0]), dim=0)
            src_mask[0] = torch.cat((src_mask[0], src_mask[0]), dim=0)
            r2l = torch.full(
                (single_direction_size, 1),
                fill_value=vocab.EOS_IDX,
                dtype=torch.long,
                device=self.device,
            )
        else:
            # Only run l2r beam search
            batch_size = single_direction_size
            r2l_start_idx = single_direction_size

        batch_beam_size = batch_size * beam_size

        input_ids = torch.full(
            (single_direction_size, 1),
            fill_value=vocab.SOS_IDX,
            dtype=torch.long,
            device=self.device,
        )
        # instantiate new beam managers

        if bi_dir:
            input_ids = torch.cat((input_ids, r2l), dim=0)

        beam_scorer = BeamSearchScorer(
            batch_size, beam_size, alpha, early_stopping, self.device
        )

        # first beam search
        hyps, scores = self._beam_search(
            src=src,
            src_mask=src_mask,
            input_ids=input_ids,
            beam_scorer=beam_scorer,
            beam_size=beam_size,
            max_len=max_len,
            temperature=temperature,
        )

        # reverse last half, if we are running both l2r/r2l.
        if bi_dir:
            for i in range(r2l_start_idx, batch_beam_size):
                hyps[i] = torch.flip(hyps[i], dims=[0])

        # Now we can guarantee that every hyp is in "l2r" form.

        if scoring_run and bi_dir:
            # compute the final score, by using the reversed sequence as target/out
            # and calculating the final score by using a modified loss from this target
            # thus, we score the model by using the opposite direction as target

            # plus to append start token
            max_len = max([len(h) + 1 for h in hyps])

            # start with l2r reversing
            r2l_tgt, r2l_out = to_tgt_output(
                hyps[:r2l_start_idx], "r2l", self.device, pad_to_len=max_len
            )
            l2r_tgt, l2r_out = to_tgt_output(
                hyps[r2l_start_idx:], "l2r", self.device, pad_to_len=max_len
            )
            tgt = torch.cat((l2r_tgt, r2l_tgt), dim=0)
            out = torch.cat((l2r_out, r2l_out), dim=0)

            # calculate final score, by passing
            rev_scores = self._rate(src, src_mask, tgt, out, alpha, temperature)
            if bi_dir:
                # flip order of the scores from [l2r r2l] -> [r2l l2r]
                rev_scores = torch.cat(
                    (rev_scores[r2l_start_idx:], rev_scores[:r2l_start_idx]), dim=0
                )
            scores = scores + rev_scores
        # Scores dim: [(2 *, if bi_dir )b * beam_size]
        # Goal now: Rearrange to [b, (2 *, if bi_dir )beam_size], s.t. we can choose the best ones from either direction
        scores = rearrange(scores, "(b m) -> b m", b=batch_size)
        if bi_dir:
            # After rearrange: [2 * b, beam_size]
            l2r_scores, r2l_scores = torch.chunk(scores, 2, dim=0)
            # conact both splits in the second dim to get: [b, 2 * beam_size]
            scores = torch.cat((l2r_scores, r2l_scores), dim=1)

        # best_scores contains the max score, indices the idx into the [2 * beam_size] dim, for each batch
        # (best_scores: [batch_size,], best_indices: [batch_size,])
        best_scores, best_indices = torch.max(scores, dim=1)
        # calculate the originating hyp index, s.t. we can find the corresponding hyp based on the
        # index from the re-arranged [2 * beam_size] scores.

        if bi_dir:
            # if the idx is beyond the normal beam_size, it points to a r2l hyp
            # we need to save this offset, to later add it to the final hyp index
            r2l_split_offsets = torch.where(best_indices >= beam_size, r2l_start_idx, 0)
        # calculate the originating hyp index.
        best_indices = (
                (
                    # first, add the offset where all k-best beams for a single batch starts
                    #   remember, the layout is
                    #   [ m * batch_0,..., m * batch_n || m * batch_0, ..., m * batch_n]
                    #               l2r                             r2l
                        torch.arange(
                            0, single_direction_size, dtype=torch.long, device=self.device
                        ) * beam_size
                ) + (
                    # second, add the actual beam offset
                    best_indices % beam_size
                )
        )

        if bi_dir:
            # third, if we originate from the r2l split, we need to add the offset
            best_indices = best_indices + r2l_split_offsets

        # Now, use the hyp-indices with the potentially corrected score to return the single-best hyps for each
        # request based on the k-best beam search results.
        return [Hypothesis(hyps[idx], score, "l2r") for (idx, score) in zip(best_indices, best_scores)]

    def _beam_search(
            self,
            src: List[FloatTensor],
            src_mask: List[LongTensor],
            input_ids: LongTensor,
            beam_scorer: BeamSearchScorer,
            beam_size: int,
            max_len: int,
            temperature: float,
    ) -> Tuple[List[LongTensor], FloatTensor]:
        """inner beam search

        Parameters
        ----------
        src : List[FloatTensor]
            [b, t, d]
        src_mask : List[LongTensor]
            [b, t]
        input_ids: LongTensor
            [b, 1]
        beam_size : int
        max_len : int

        Returns
        _______
        Tuple[List[LongTensor], FloatTensor]
            List[LongTensor]: [b * beam_size] without SOS or EOS token
            FloatTensor: [b * beam_size] corresponding scores
        """
        batch_size, cur_len = input_ids.shape

        beam_scores = torch.zeros(batch_size, dtype=torch.float, device=self.device)

        while cur_len < max_len and not beam_scorer.is_done():
            next_token_logits = (
                    self.transform(src, src_mask, input_ids)[:, -1, :] / temperature
            )
            # [b *, l, v]
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)

            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(
                next_token_scores
            )
            # [batch_size, beam_size * vocab_size]
            reshape_size = next_token_scores.shape[0] // batch_size
            next_token_scores = rearrange(
                next_token_scores,
                "(b m) v -> b (m v)",
                m=reshape_size,
            )

            # [b, 2 * beam_size]
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * beam_size, dim=1
            )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode='trunc')
            next_tokens = next_tokens % vocab_size

            if cur_len == 1:
                input_ids = repeat(input_ids, "b l -> (b m) l", m=beam_size)
                for i in range(len(src)):
                    src[i] = repeat(src[i], "b ... -> (b m) ...", m=beam_size)
                    src_mask[i] = repeat(src_mask[i], "b ... -> (b m) ...", m=beam_size)

            beam_scores, beam_next_tokens, beam_idx = beam_scorer.process(
                input_ids=input_ids,
                next_scores=next_token_scores,
                next_tokens=next_tokens,
                next_indices=next_indices,
            )

            input_ids = torch.cat(
                (input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)), dim=-1
            )

            cur_len += 1

        return beam_scorer.finalize(input_ids, beam_scores)

    def _rate(
            self,
            src: List[FloatTensor],
            src_mask: List[LongTensor],
            tgt: LongTensor,
            out: LongTensor,
            alpha: float,
            temperature: float,
    ) -> FloatTensor:
        """rate tgt and output

        Parameters
        ----------
        src : List[FloatTensor]
            [b * beam_size, t, d]
        src_mask : List[LongTensor]
            [b * beam_size, t]
        tgt : LongTensor
            [b * beam_size, l]
        out : LongTensor
            [b * beam_size, l]
        alpha : float
        temperature : float

        Returns
        -------
        FloatTensor
            [b * beam_size]
        """
        b = tgt.shape[0]
        out_hat = self.transform(src, src_mask, tgt) / temperature

        loss = ce_loss(out_hat, out, reduction="none")
        loss = rearrange(loss, "(b l) -> b l", b=b)

        mask = tgt == vocab.PAD_IDX
        penalty = (~mask).sum(dim=1) ** alpha
        loss = -torch.sum(loss, dim=1) / penalty

        return loss

    def _new_rate(
            self,
            src: FloatTensor,
            src_mask: LongTensor,
            tgt: LongTensor,
            out: LongTensor,
            alpha: float,
            temperature: float,
    ) -> FloatTensor:
        """rate tgt and output

        Parameters
        ----------
        src : List[FloatTensor]
            [b * beam_size, t, d]
        src_mask : List[LongTensor]
            [b * beam_size, t]
        tgt : LongTensor
            [b * beam_size, l]
        out : LongTensor
            [b * beam_size, l]
        alpha : float
        temperature : float

        Returns
        -------
        FloatTensor
            [b * beam_size]
        """
        b = tgt.shape[0]
        out_hat = self(src, src_mask, tgt) / temperature

        loss = ce_loss(out_hat, out, reduction="none")
        loss = rearrange(loss, "(b l) -> b l", b=b)

        mask = tgt == vocab.PAD_IDX
        penalty = (~mask).sum(dim=1) ** alpha
        loss = -torch.sum(loss, dim=1) / penalty

        return loss

    def new_beam_search(
            self,
            src: FloatTensor,
            src_mask: LongTensor,
            beam_size: int,
            max_len: int,
            alpha: float,
            early_stopping: bool,
            temperature: float,
            bi_dir: bool = True,
            scoring_run: bool = True,
    ) -> List[Hypothesis]:
        """run beam search to decode
        Parameters
        ----------
        src : List[FloatTensor]
            [b, h, w, d]
            List, because _beam_search function mutates the tensor directly
            It could return the mutated new_src however...
        src_mask : List[LongTensor]
            [b, h, w]
            List, because _beam_search function mutates the tensor directly
            It could return the mutated new_src_mask however...
        beam_size : int
        max_len : int
        alpha : float
        early_stopping : bool

        Returns
        -------
        List[Hypothesis]: [batch_size,]
        """
        # If we want to run bi-directional beam_search, we generate double the hypothesis.
        # We aim to generate k-best Hyps for each - L2R and R2L - decoding.
        # Thus, we have to double the batch, s.t. we start with
        # (batch * 2 * beam_size) Beams which we explore to generate (batch * 2 * beam_size) Hyps
        batch_size = src.shape[0]

        # instantiate new beam managers.
        # Each Beam Manager analyzes the results from a single prediction step
        # and updates its state: creating new beams, pruning beams, being done, ...
        beam_managers = [
            BeamManager(
                max_beams=beam_size,
                src_idx=src_idx,
                is_l2r=True,
                device=self.device,
                max_len=max_len,
            ) for src_idx in range(batch_size)
        ]

        if bi_dir:
            beam_managers = beam_managers + [
                BeamManager(
                    max_beams=beam_size,
                    src_idx=src_idx,
                    is_l2r=False,
                    device=self.device,
                    max_len=max_len,
                ) for src_idx in range(batch_size)
            ]

        while True:
            next_src, next_src_mask, next_input_ids, bm_refs = generate_next_inputs(
                src, src_mask, beam_managers, device=self.device
            )
            if len(bm_refs) == 0:
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
                    self(next_src, next_src_mask, next_input_ids)[:, -1, :] / temperature,
                    dim=-1
                ),
                k=beam_size
            )
            current_beam_idx = 0
            for (bm_ref, amount_active_beams) in bm_refs:
                next_beam_idx = current_beam_idx + amount_active_beams
                beam_managers[bm_ref].update(topk_for_active_beams_scores[current_beam_idx:next_beam_idx, :],
                                             topk_for_active_beams_indices[current_beam_idx:next_beam_idx, :])
                current_beam_idx = next_beam_idx
        hyps: List[LongTensor] = []
        scores: Union[Tensor | List[Tensor]] = []
        for bm in beam_managers:
            score, hyp = bm.get_best_l2r_finalized()
            scores.append(score.unsqueeze(0))
            hyps.append(hyp)
        scores = torch.cat(scores)

        if scoring_run and bi_dir:
            # compute the final score, by using the reversed sequence as target/out
            # and calculating the final score by using a modified loss from this target
            # thus, we score the model by using the opposite direction as target

            # plus to append start token

            max_len = max([len(h) + 1 for h in hyps])

            # start with l2r reversing
            r2l_tgt, r2l_out = to_tgt_output(
                hyps[:batch_size], "r2l", self.device, pad_to_len=max_len
            )
            l2r_tgt, l2r_out = to_tgt_output(
                hyps[batch_size:], "l2r", self.device, pad_to_len=max_len
            )
            tgt = torch.cat((l2r_tgt, r2l_tgt), dim=0)
            out = torch.cat((l2r_out, r2l_out), dim=0)

            # calculate final score, by passing
            rev_scores = self._new_rate(
                torch.cat((src, src)),
                torch.cat((src_mask, src_mask)),
                tgt, out, alpha, temperature
            )
            # flip order of the scores from [l2r r2l] -> [r2l l2r]
            rev_scores = torch.cat(
                (rev_scores[batch_size:], rev_scores[:batch_size]), dim=0
            )

            scores = scores + rev_scores
        if bi_dir:
            # Choose either the l2r or the r2l hypothesis, based on score
            for i in range(batch_size):
                if scores[i] < scores[batch_size + i]:
                    scores[i] = scores[batch_size + i]
                    hyps[i] = hyps[batch_size + i]

        # Now, use the hyp-indices with the potentially corrected score to return the single-best hyps for each
        # request based on the k-best beam search results.
        return [Hypothesis(hyp, score, "l2r") for (hyp, score) in zip(hyps[:batch_size], scores[:batch_size])]

def generate_next_inputs(
        src: FloatTensor,
        src_mask: LongTensor,
        beam_managers: List[BeamManager],
        device: torch.device
):
    """Helper function to generate the model inputs based on the current beams
        Parameters
        ----------
        src : FloatTensor
            [b, t, d]
        src_mask : LongTensor
            [b, t]
        beam_managers : List[BeamManager]
        device : torch.device       where to create the returned tensors

        Returns
        _______
        Tuple[
            FloatTensor, # [ (variable-active-beams), t, d ] next_src, repeats src[i] if a beam manager for this batch-entry is active
            FloatTensor, # [ (variable-active-beams), t, d ] next_src, repeats src[i] if a beam manager for this batch-entry is active
            LongTensor,
       ]
            List[LongTensor]: [b * beam_size] without SOS or EOS token
            FloatTensor: [b * beam_size] corresponding scores
        """
    repeats = torch.zeros((src.shape[0],), dtype=torch.int, device=device)
    current_ids: List[List[LongTensor]] = [[] for _ in iter.repeat(None, src.shape[0])]
    bm_refs: List[List[Tuple[int, int]]] = [[] for _ in iter.repeat(None, src.shape[0])]

    empty = True
    for idx, bm in enumerate(beam_managers):
        active_beams = bm.active_beams.shape[0]
        if active_beams > 0:
            empty = False
            repeats[bm.src_idx] = repeats[bm.src_idx] + active_beams
            current_ids[bm.src_idx].append(bm.active_beams)
            bm_refs[bm.src_idx].append((idx, active_beams))

    if empty:
        return torch.empty((0,), device=device), torch.empty((0,), device=device), torch.empty((0,), device=device), []

    return (
        torch.repeat_interleave(src, repeats, dim=0),
        torch.repeat_interleave(src_mask, repeats, dim=0),
        torch.cat(list(iter.chain.from_iterable(current_ids)), dim=0),
        list(iter.chain.from_iterable(bm_refs))
    )
