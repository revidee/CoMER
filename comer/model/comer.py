from typing import List, Union

import pytorch_lightning as pl
import torch
from torch import FloatTensor, LongTensor

from comer.utils.utils import Hypothesis

from .decoder import Decoder
from .encoder import Encoder


class CoMER(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        growth_rate: int,
        num_layers: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
    ):
        super().__init__()

        self.encoder = Encoder(
            d_model=d_model, growth_rate=growth_rate, num_layers=num_layers
        )
        self.decoder = Decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
        )

    def forward(
        self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor,
            l2r_indices: Union[LongTensor, None] = None, r2l_indices: Union[LongTensor, None] = None
    ) -> FloatTensor:
        """run img and bi-tgt

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h, w]
        img_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [2b, l]

        Returns
        -------
        FloatTensor
            [2b, l, vocab_size]
        """
        feature, mask = self.encoder(img, img_mask)  # [b, t, d]
        if l2r_indices is None or r2l_indices is None:
            feature = torch.cat((feature, feature), dim=0)  # [2b, t, d]
            mask = torch.cat((mask, mask), dim=0)
        else:
            feature = torch.cat(
                (
                    torch.repeat_interleave(feature, l2r_indices, dim=0),
                    torch.repeat_interleave(feature, r2l_indices, dim=0)
                ), dim=0)  # [2b, t, d]
            mask = torch.cat((
                torch.repeat_interleave(mask, l2r_indices, dim=0),
                torch.repeat_interleave(mask, r2l_indices, dim=0)
            ), dim=0)

        out = self.decoder(feature, mask, tgt)

        return out

    def beam_search(
            self,
            img: FloatTensor,
            img_mask: LongTensor,
            beam_size: int,
            max_len: int,
            alpha: float,
            early_stopping: bool,
            temperature: float,
            bi_dir: bool = True,
            scoring_run: bool = True,
            debug: bool = False,
            **kwargs,
    ) -> List[Hypothesis]:
        """run bi-direction beam search for given img

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h', w']
        img_mask: LongTensor
            [b, h', w']
        beam_size : int
        max_len : int

        Returns
        -------
        List[Hypothesis]
        """
        feature, mask = self.encoder(img, img_mask)  # [b, t, d]
        return self.decoder.beam_search(
            [feature], [mask], beam_size, max_len, alpha, early_stopping, temperature,
            bi_dir=bi_dir, scoring_run=scoring_run, debug=debug
        )

    def new_beam_search(
            self,
            img: FloatTensor,
            img_mask: LongTensor,
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
            global_pruning: str = 'none',
            **kwargs,
    ) -> List[Hypothesis]:
        """run bi-direction beam search for given img

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h', w']
        img_mask: LongTensor
            [b, h', w']
        beam_size : int
        max_len : int

        Returns
        -------
        List[Hypothesis]
        """
        feature, mask = self.encoder(img, img_mask)  # [b, t, d]
        return self.decoder.new_beam_search(
            feature, mask, beam_size, max_len, alpha, early_stopping,
            temperature, bi_dir=bi_dir, scoring_run=scoring_run, debug=debug, save_logits=save_logits,
            logit_norm_temp=logit_norm_temp, global_pruning=global_pruning
        )
