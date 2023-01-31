from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from comer.datamodules.crohme import vocab
from einops import rearrange
from torch import LongTensor, FloatTensor, Tensor
from torch import linalg as LA
from torchmetrics import Metric

from comer.datamodules.crohme.batch import MaybePartialLabelWithIndices


class Hypothesis:
    seq: List[int]
    score: float
    history: List[float]
    was_l2r: bool

    all_l2r_hyps: Union[List[LongTensor], None] = None
    all_l2r_scores: Union[FloatTensor, None] = None
    all_l2r_history: Union[List[FloatTensor], None] = None
    all_r2l_hyps: Union[List[LongTensor], None] = None
    all_r2l_scores: Union[FloatTensor, None] = None
    all_r2l_history: Union[List[FloatTensor], None] = None

    best_rev: List[float]
    all_l2r_rev_scores: Union[List[FloatTensor], None] = None
    all_r2l_rev_scores: Union[List[FloatTensor], None] = None

    def __init__(
            self,
            seq_tensor: LongTensor,
            score: float,
            direction: str,
            history: Union[FloatTensor, None] = None,
            was_l2r: bool = False,
            all_l2r_hyps: Union[List[LongTensor], None] = None,
            all_l2r_scores: Union[FloatTensor, None] = None,
            all_l2r_history: Union[List[FloatTensor], None] = None,
            all_r2l_hyps: Union[List[LongTensor], None] = None,
            all_r2l_scores: Union[FloatTensor, None] = None,
            all_r2l_history: Union[List[FloatTensor], None] = None,
            best_rev: Union[FloatTensor, None] = None,
            all_l2r_rev_scores: Union[List[FloatTensor], None] = None,
            all_r2l_rev_scores: Union[List[FloatTensor], None] = None,
            raw_logits: Union[FloatTensor, None] = None,
            raw_logits_rev: Union[FloatTensor, None] = None
    ) -> None:
        assert direction in {"l2r", "r2l"}
        raw_seq = seq_tensor.tolist()

        if direction == "r2l":
            result = raw_seq[::-1]
        else:
            result = raw_seq

        self.seq = result
        self.score = score

        self.was_l2r = was_l2r

        if history is not None:
            self.history = history.tolist()
        else:
            self.history = []

        self.all_l2r_hyps = all_l2r_hyps
        self.all_l2r_scores = all_l2r_scores
        self.all_l2r_history = all_l2r_history

        self.all_r2l_hyps = all_r2l_hyps
        self.all_r2l_scores = all_r2l_scores
        self.all_r2l_history = all_r2l_history

        if best_rev is not None:
            self.best_rev = best_rev.tolist()
        else:
            self.best_rev = []

        self.all_l2r_rev_scores = all_l2r_rev_scores
        self.all_r2l_rev_scores = all_r2l_rev_scores

        self.raw_logits = raw_logits
        self.raw_logits_rev = raw_logits_rev

    def __len__(self):
        if len(self.seq) != 0:
            return len(self.seq)
        else:
            return 1

    def __str__(self):
        return f"seq: {self.seq}, score: {self.score}"


class ExpRateRecorder(Metric):
    full_state_update = False

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("total_line", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("rec", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, indices_hat: List[List[int]], indices: List[List[int]]):
        for pred, truth in zip(indices_hat, indices):
            if pred == truth:
                self.rec += 1

            self.total_line += 1

    def compute(self) -> float:
        exp_rate = self.rec / self.total_line
        return exp_rate


def ce_loss(
        output_hat: torch.Tensor,
        output: torch.Tensor,
        ignore_idx: int = vocab.PAD_IDX,
        reduction: str = "mean",
) -> torch.Tensor:
    """comput cross-entropy loss

    Args:
        output_hat (torch.Tensor): [batch, len, e]
        output (torch.Tensor): [batch, len]
        ignore_idx (int):

    Returns:
        torch.Tensor: loss value
    """
    flat_hat = rearrange(output_hat, "b l e -> (b l) e")
    flat = rearrange(output, "b l -> (b l)")
    loss = F.cross_entropy(flat_hat, flat, ignore_index=ignore_idx, reduction=reduction)
    return loss

def ce_logitnorm_loss(
        output_hat: torch.Tensor,
        output: torch.Tensor,
        temperature: torch.Tensor,
        ignore_idx: int = vocab.PAD_IDX,
        reduction: str = "mean",
) -> torch.Tensor:
    """comput cross-entropy loss with logit normalization from https://arxiv.org/abs/2205.09310

    Args:
        output_hat (torch.Tensor): [batch, len, e]
        output (torch.Tensor): [batch, len]
        temperature: (torch.Tensor), (1) the T parameter from the paper
        ignore_idx (int):

    Returns:
        torch.Tensor: loss value
    """
    mod_hat = output_hat / (temperature * (LA.vector_norm(output_hat, dim=-1, keepdim=True) + 1e-7))
    flat_hat = rearrange(mod_hat, "b l e -> (b l) e")
    flat = rearrange(output, "b l -> (b l)")
    loss = F.cross_entropy(flat_hat, flat, ignore_index=ignore_idx, reduction=reduction)
    return loss


def to_tgt_output(
        batched_tokens: Union[List[List[int]], List[LongTensor]],
        direction: str,
        device: torch.device,
        pad_to_len: Optional[int] = None,
) -> Tuple[LongTensor, LongTensor]:
    """Generate tgt and out for indices

    Parameters
    ----------
    batched_tokens : Union[List[List[int]], List[LongTensor]]
        indices: [b, l]
    direction : str
        one of "l2f" and "r2l"
    device : torch.device

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        tgt, out: [b, l], [b, l]
    """
    assert direction in {"l2r", "r2l"}

    if isinstance(batched_tokens[0], list):
        batched_tokens = [torch.tensor(single_tokens, dtype=torch.long) for single_tokens in batched_tokens]

    if direction == "l2r":
        start_w = vocab.SOS_IDX
        stop_w = vocab.EOS_IDX
    else:
        batched_tokens = [torch.flip(single_tokens, dims=[0]) for single_tokens in batched_tokens]
        start_w = vocab.EOS_IDX
        stop_w = vocab.SOS_IDX

    batch_size = len(batched_tokens)
    lens = [len(t) for t in batched_tokens]

    length = max(lens) + 1
    if pad_to_len is not None:
        length = max(length, pad_to_len)

    tgt = torch.full(
        (batch_size, length),
        fill_value=vocab.PAD_IDX,
        dtype=torch.long,
        device=device,
    )
    out = torch.full(
        (batch_size, length),
        fill_value=vocab.PAD_IDX,
        dtype=torch.long,
        device=device,
    )

    # Fill the Teacher-Forcing "target" tensor and the Ouput "target" tensor which is used to compute the loss.
    for i, token in enumerate(batched_tokens):
        # teacher forcing input: <SOS> <token1> <token2> <...> <tokenN> <optional-pad> <optional-pad> (l2r)
        tgt[i, 0] = start_w
        tgt[i, 1: (1 + lens[i])] = token

        # expected model output: <token1> <...> <tokenN> <EOS> <optional-pad> <optional-pad> (l2r)
        out[i, : lens[i]] = token
        out[i, lens[i]] = stop_w

    return tgt, out


def to_tgt_output_partial(
        maybe_partials: List[MaybePartialLabelWithIndices],
        direction: str,
        device: torch.device
) -> Tuple[LongTensor, LongTensor, LongTensor]:
    """Generate tgt and out for indices

    Parameters
    ----------
    maybe_partials : List[MaybePartialLabelWithIndices]
        indices: [b, l]
    direction : str
        one of "l2r" and "r2l"
    device : torch.device

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, List[int]]
        tgt, out: [b', l], ['b, l]
        src_indices: [b']
        b': Non-Zero Length batch entries.
        Entries can be blank if a partial hypothesis only contains a L2R but no R2L hyp.
    """
    assert direction in {"l2r", "r2l"}

    if direction == "l2r":
        start_w = vocab.SOS_IDX
        stop_w = vocab.EOS_IDX
    else:

        start_w = vocab.EOS_IDX
        stop_w = vocab.SOS_IDX

    token_tensors: List[Tensor] = []
    token_tensors_append = token_tensors.append

    srcs_with_labels: List[int] = []
    srcs_with_labels_append = srcs_with_labels.append

    src_repeats = torch.ones((len(maybe_partials),), dtype=torch.long, device=device)

    max_len = 0

    if direction == "l2r":
        for i, label_tuple in enumerate(maybe_partials):
            len_l2r = len(label_tuple[1]) if label_tuple[1] is not None else 0
            len_r2l = len(label_tuple[2]) if label_tuple[2] is not None else 0
            max_len_bi_partial = len_l2r if len_l2r > len_r2l else len_r2l
            max_len = max_len if max_len_bi_partial < max_len else max_len_bi_partial
            if not label_tuple[0] or (label_tuple[1] is not None and len(label_tuple[1]) > 0):
                token_tensors_append(torch.tensor(label_tuple[1], dtype=torch.long, device=device))
                srcs_with_labels_append(i)
            else:
                src_repeats[i] = 0

    else:
        for i, label_tuple in enumerate(maybe_partials):
            len_l2r = len(label_tuple[1]) if label_tuple[1] is not None else 0
            len_r2l = len(label_tuple[2]) if label_tuple[2] is not None else 0
            max_len_bi_partial = len_l2r if len_l2r > len_r2l else len_r2l
            max_len = max_len if max_len_bi_partial < max_len else max_len_bi_partial
            if not label_tuple[0] or (label_tuple[2] is not None and len(label_tuple[2]) > 0):
                tokens = label_tuple[2] if label_tuple[0] else label_tuple[1]
                token_tensors_append(torch.flip(torch.tensor(tokens, dtype=torch.long, device=device), dims=[0]))
                srcs_with_labels_append(i)
            else:
                src_repeats[i] = 0

    batch_size = len(token_tensors)

    tgt = torch.full(
        (batch_size, max_len + 1),
        fill_value=vocab.PAD_IDX,
        dtype=torch.long,
        device=device,
    )
    out = torch.full(
        (batch_size, max_len + 1),
        fill_value=vocab.PAD_IDX,
        dtype=torch.long,
        device=device,
    )

    # Fill the Teacher-Forcing "target" tensor and the Ouput "target" tensor which is used to compute the loss.
    for i, (tokens, src_idx) in enumerate(zip(token_tensors, srcs_with_labels)):
        # teacher forcing input: <SOS> <token1> <token2> <...> <tokenN> <optional-pad> <optional-pad> (l2r)
        tgt[i, 0] = start_w
        if maybe_partials[src_idx][0]:
            # is partial
            tgt[i, 1: (tokens.size(0))] = tokens[:-1]
            # expected model output: <token1> <...> <tokenN> <optional-pad> <optional-pad> <optional-pad> (l2r)
            # here, we don't have an EOS, since it is a partial label
            out[i, : tokens.size(0)] = tokens
        else:
            # is not partial
            tgt[i, 1: (1 + tokens.size(0))] = tokens
            # expected model output: <token1> <...> <tokenN> <EOS> <optional-pad> <optional-pad> (l2r)
            out[i, : tokens.size(0)] = tokens
            out[i, tokens.size(0)] = stop_w

    return tgt, out, src_repeats


def to_bi_tgt_out(
        tokens: List[MaybePartialLabelWithIndices], device: torch.device,
) -> Tuple[LongTensor, LongTensor, LongTensor, LongTensor]:
    """Generate bidirection tgt and out

    Parameters
    ----------
    tokens : List[List[int]]
        indices: [b, l]
    device : torch.device

    Returns
    -------
    Tuple[LongTensor, LongTensor, List[int], List[int]]
        tgt, out: [b_L + b_R, l], [b_L + b_R, l]
        l2r_indices: [b] - indices, s.t. a torch.repeat_interleave results in b_L length
        r2l_indices: [b] - indices, s.t. a torch.repeat_interleave results in b_R length
        b_L: Non-Zero Length batch entries for L2R Hypothesis.
        b_R: Non-Zero Length batch entries for R2L Hypothesis.
    """
    l2r_tgt, l2r_out, l2r_indices = to_tgt_output_partial(tokens, "l2r", device)
    r2l_tgt, r2l_out, r2l_indices = to_tgt_output_partial(tokens, "r2l", device)
    tgt = torch.cat((l2r_tgt, r2l_tgt), dim=0)
    out = torch.cat((l2r_out, r2l_out), dim=0)

    return tgt, out, l2r_indices, r2l_indices
