from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from comer.datamodules.crohme import vocab
from einops import rearrange
from torch import LongTensor, FloatTensor
from torchmetrics import Metric


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
            all_r2l_history: Union[List[FloatTensor], None] = None
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
            pred = vocab.indices2label(pred)
            truth = vocab.indices2label(truth)

            is_same = pred == truth

            if is_same:
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


def to_bi_tgt_out(
        tokens: List[List[int]], device: torch.device,
) -> Tuple[LongTensor, LongTensor]:
    """Generate bidirection tgt and out

    Parameters
    ----------
    tokens : List[List[int]]
        indices: [b, l]
    device : torch.device

    Returns
    -------
    Tuple[LongTensor, LongTensor]
        tgt, out: [2b, l], [2b, l]
    """
    l2r_tgt, l2r_out = to_tgt_output(tokens, "l2r", device)
    r2l_tgt, r2l_out = to_tgt_output(tokens, "r2l", device)

    tgt = torch.cat((l2r_tgt, r2l_tgt), dim=0)
    out = torch.cat((l2r_out, r2l_out), dim=0)

    return tgt, out
