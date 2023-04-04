from abc import abstractmethod
from typing import List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops import rearrange
from einops.einops import repeat
from torch import FloatTensor, LongTensor

from comer.datamodules.crohme import vocab
from comer.utils.beam_search import BatchedBeamSearch, invalid_score
from comer.utils.original_beam_search import BeamSearchScorer
from comer.utils.utils import Hypothesis, ce_loss, to_tgt_output


# modified from
# https://github.com/huggingface/transformers/blob/af6e01c5bc39467f1e3ce47a2135fb1777af1db2/src/transformers/generation_utils.py#L1843


class DecodeModel(pl.LightningModule):

    def __init__(self, used_vocab=vocab):
        super().__init__()
        self.vocab = used_vocab


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
            debug: bool = False
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
                fill_value=self.vocab.EOS_IDX,
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
            fill_value=self.vocab.SOS_IDX,
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

        if debug:
            print("score before")
            for i, hyp in enumerate(hyps):
                print(f"{scores[i]:.4f}", self.vocab.indices2words(hyp.tolist()))

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
            if debug:
                print("rev_scores:")
                for i, single_tgt in enumerate(tgt):
                    if i < r2l_start_idx:
                        idx = i + r2l_start_idx
                    else:
                        idx = i - r2l_start_idx
                    print(f"{rev_scores[i]:.4f}", self.vocab.indices2words(tgt[idx].tolist()), "for",
                          self.vocab.indices2words(hyps[i].tolist()))
            scores = scores + rev_scores
        if debug:
            print("score after")
            for i, hyp in enumerate(hyps):
                print(f"{scores[i]:.4f}", self.vocab.indices2words(hyp.tolist()))
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

            next_indices = torch.div(next_tokens, len(self.vocab), rounding_mode='trunc')
            next_tokens = next_tokens % len(self.vocab)

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

        mask = tgt == self.vocab.PAD_IDX
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
        unsharpened = self(src, src_mask, tgt)
        out_hat = unsharpened / temperature

        loss = ce_loss(out_hat, out, reduction="none")
        loss = rearrange(loss, "(b l) -> b l", b=b)

        mask = tgt == self.vocab.PAD_IDX
        penalty = (~mask).sum(dim=1) ** alpha
        loss = -torch.sum(loss, dim=1) / penalty

        return loss, unsharpened

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
            debug: bool = False,
            save_logits: bool = False,
            logit_norm_temp: float = -1.,
            global_pruning: str = 'none', # none, st, sup
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
        GLOBAL_PRUNING_THRESHOLDS_FOR_EPOCHS_PRESETS = {
            'none': [],
            'sup': [(15, 0.8), (30, 0.4), (60, 0.1), (400, 0.05)],
            'sup_hme': [(2, 0.8), (10, 0.4), (15, 0.1), (25, 0.05)],
            'st': [(30, 0.1), (400, 0.05)],
            'st2': [(30, 0.3), (400, 0.15)],
            'partial': [(400, 0.05)],
        }
        assert global_pruning in GLOBAL_PRUNING_THRESHOLDS_FOR_EPOCHS_PRESETS

        global_pruning_threshold = invalid_score
        for (epoch, threshold) in GLOBAL_PRUNING_THRESHOLDS_FOR_EPOCHS_PRESETS[global_pruning]:
            if self.current_epoch <= epoch:
                global_pruning_threshold = threshold
                break

        beamsearch = BatchedBeamSearch(
            beam_size,
            self.device,
            max_len,
            bi_dir,
            temperature=temperature,
            find_top_k=beam_size * 2,
            debug=debug,
            save_logits=save_logits,
            logit_norm_temp=logit_norm_temp,
            min_normalized_pseudo_probabilty=global_pruning_threshold,
            used_vocab=self.vocab
        )
        if save_logits:
            hyps_l2r, history_l2r, scores_l2r, repeats_l2r, raw_logits_l2r, \
                hyps_r2l, history_r2l, scores_r2l, repeats_r2l, raw_logits_r2l = beamsearch.predict(self, src, src_mask)
        else:
            hyps_l2r, history_l2r, scores_l2r, repeats_l2r, \
                hyps_r2l, history_r2l, scores_r2l, repeats_r2l = beamsearch.predict(self, src, src_mask)

        hyps_l2r_len = len(hyps_l2r)
        hyps_rl2_len = len(hyps_r2l)
        if scoring_run and bi_dir and (hyps_l2r_len + hyps_rl2_len) > 0:
            # compute the final score, by using the reversed sequence as target/out
            # and adding a custom loss to the score.
            # thus, we re-score the hyps by using the opposite direction as target

            # plus to append start token
            max_l2r = max([len(h) for h in hyps_l2r]) if hyps_l2r_len > 0 else 0
            max_r2l = max([len(h) for h in hyps_r2l]) if len(hyps_r2l) > 0 else 0

            max_len = max((max_l2r, max_r2l)) + 1

            # start with l2r reversing
            if hyps_l2r_len > 0:
                r2l_tgt, r2l_out = to_tgt_output(
                    hyps_l2r, "r2l", self.device, pad_to_len=max_len
                )
                tgt, out = r2l_tgt, r2l_out
            if hyps_rl2_len > 0:
                l2r_tgt, l2r_out = to_tgt_output(
                    hyps_r2l, "l2r", self.device, pad_to_len=max_len
                )
                tgt, out = l2r_tgt, l2r_out
            if hyps_rl2_len > 0 and hyps_l2r_len > 0:
                tgt = torch.cat((r2l_tgt, l2r_tgt), dim=0)
                out = torch.cat((r2l_out, l2r_out), dim=0)

            rev_scores, raw_rev_logits = self._new_rate(
                torch.cat(
                    (
                        torch.repeat_interleave(src, repeats_l2r, dim=0),
                        torch.repeat_interleave(src, repeats_r2l, dim=0),
                    ), dim=0
                ),
                torch.cat(
                    (
                        torch.repeat_interleave(src_mask, repeats_l2r, dim=0),
                        torch.repeat_interleave(src_mask, repeats_r2l, dim=0),
                    ), dim=0
                ),
                tgt, out, alpha, temperature
            )
            out_hat = F.log_softmax(
                raw_rev_logits / temperature,
                dim=-1
            )
            if hyps_l2r_len > 0:
                out_hat[:hyps_l2r_len] = torch.flip(
                    out_hat[:hyps_l2r_len],
                    dims=[1]
                )
            if debug:
                print("rev_scores:")
                for i, single_tgt in enumerate(tgt):
                    if i < hyps_l2r_len:
                        print(f"{rev_scores[i]:.4f}", self.vocab.indices2words(single_tgt.tolist()), "for (l2r)",
                              self.vocab.indices2words(hyps_l2r[i].tolist()))
                    else:
                        print(f"{rev_scores[i]:.4f}", self.vocab.indices2words(single_tgt.tolist()), "for (r2l)",
                              self.vocab.indices2words(hyps_r2l[i - hyps_l2r_len].tolist()))

            scores_l2r += rev_scores[:hyps_l2r_len]
            scores_r2l += rev_scores[hyps_l2r_len:]
        output_hyps: List[Hypothesis] = []
        curr_offset_l2r = 0
        curr_offset_r2l = 0
        l2r_rev: List[FloatTensor] = []
        r2l_rev: List[FloatTensor] = []
        l2r_rev_raw_logits: List[FloatTensor] = []
        r2l_rev_raw_logits: List[FloatTensor] = []
        # Find the best hyp from the (optionally rescored) set of best l2r/r2l hyps.
        for src_idx, (len_l2r, len_r2l) in enumerate(zip(repeats_l2r, repeats_r2l)):
            # choose the best candidate for each input from the batch
            len_l2r = len_l2r.item()
            len_r2l = len_r2l.item()
            curr_best_idx = -1
            curr_best_idx_from_l2r = True
            curr_best_score = float('-Inf')
            for i in range(len_l2r):
                hyp_cand_idx = curr_offset_l2r + i
                rev = torch.gather(
                    out_hat[
                    hyp_cand_idx,
                    (max_len - hyps_l2r[hyp_cand_idx].size(0)):
                    ],
                    1,
                    hyps_l2r[hyp_cand_idx].unsqueeze(-1),
                ).squeeze(-1)
                hyp_cand_score = scores_l2r[hyp_cand_idx]
                # hyp_cand_score = torch.min(
                #     torch.cat((history_l2r[hyp_cand_idx], rev))
                # )
                l2r_rev.append(
                    rev
                )
                l2r_rev_raw_logits.append(
                    raw_rev_logits[
                        hyp_cand_idx,
                        :hyps_l2r[hyp_cand_idx].size(0) + 1
                    ]
                )
                if hyp_cand_score > curr_best_score:
                    curr_best_idx = hyp_cand_idx
                    curr_best_score = hyp_cand_score
                    curr_best_idx_from_l2r = True
            for i in range(len_r2l):
                hyp_cand_idx = curr_offset_r2l + i
                rev = torch.gather(
                    out_hat[
                    hyps_l2r_len + hyp_cand_idx,
                    :(hyps_r2l[hyp_cand_idx].size(0))
                    ],
                    1,
                    hyps_r2l[hyp_cand_idx].unsqueeze(-1),
                ).squeeze(-1)
                hyp_cand_score = scores_r2l[hyp_cand_idx]
                # hyp_cand_score = torch.min(
                #     torch.cat((history_r2l[hyp_cand_idx], rev))
                # )
                r2l_rev.append(
                    rev
                )
                r2l_rev_raw_logits.append(
                    raw_rev_logits[
                        hyps_l2r_len + hyp_cand_idx,
                        :hyps_r2l[hyp_cand_idx].size(0) + 1
                    ]
                )
                if hyp_cand_score > curr_best_score:
                    curr_best_idx = hyp_cand_idx
                    curr_best_score = hyp_cand_score
                    curr_best_idx_from_l2r = False

            start_l2r, start_r2l = curr_offset_l2r, curr_offset_r2l
            curr_offset_r2l += len_r2l
            curr_offset_l2r += len_l2r
            if curr_best_idx != -1:
                if curr_best_idx_from_l2r:
                    output_hyps.append(
                        Hypothesis(hyps_l2r[curr_best_idx], scores_l2r[curr_best_idx],
                                   "l2r", history=history_l2r[curr_best_idx], was_l2r=True,
                                   all_l2r_hyps=hyps_l2r[start_l2r:curr_offset_l2r],
                                   all_l2r_scores=scores_l2r[start_l2r:curr_offset_l2r],
                                   all_l2r_history=history_l2r[start_l2r:curr_offset_l2r],
                                   all_r2l_hyps=hyps_r2l[start_r2l:curr_offset_r2l],
                                   all_r2l_scores=scores_r2l[start_r2l:curr_offset_r2l],
                                   all_r2l_history=history_r2l[start_r2l:curr_offset_r2l],
                                   best_rev=l2r_rev[curr_best_idx],
                                   all_l2r_rev_scores=l2r_rev[start_l2r:curr_offset_l2r],
                                   all_r2l_rev_scores=r2l_rev[start_r2l:curr_offset_r2l],
                                   raw_logits=None if not save_logits else raw_logits_l2r[curr_best_idx],
                                   raw_logits_rev=None if not save_logits else l2r_rev_raw_logits[curr_best_idx]
                                   )
                    )
                else:
                    output_hyps.append(
                        Hypothesis(hyps_r2l[curr_best_idx], scores_r2l[curr_best_idx],
                                   "l2r", history=history_r2l[curr_best_idx], was_l2r=False,
                                   all_l2r_hyps=hyps_l2r[start_l2r:curr_offset_l2r],
                                   all_l2r_scores=scores_l2r[start_l2r:curr_offset_l2r],
                                   all_l2r_history=history_l2r[start_l2r:curr_offset_l2r],
                                   all_r2l_hyps=hyps_r2l[start_r2l:curr_offset_r2l],
                                   all_r2l_scores=scores_r2l[start_r2l:curr_offset_r2l],
                                   all_r2l_history=history_r2l[start_r2l:curr_offset_r2l],
                                   best_rev=r2l_rev[curr_best_idx],
                                   all_l2r_rev_scores=l2r_rev[start_l2r:curr_offset_l2r],
                                   all_r2l_rev_scores=r2l_rev[start_r2l:curr_offset_r2l],
                                   raw_logits=None if not save_logits else raw_logits_r2l[curr_best_idx],
                                   raw_logits_rev=None if not save_logits else r2l_rev_raw_logits[curr_best_idx]
                                   )
                    )
            else:
                output_hyps.append(Hypothesis(torch.empty(0, device=self.device, dtype=torch.long), float('-Inf'), "l2r"))
        return output_hyps
