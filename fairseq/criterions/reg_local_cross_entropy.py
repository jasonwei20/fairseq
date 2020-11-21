# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import time

import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion

import random
random.seed(0)

@register_criterion('reg_local_cross_entropy')
class RegLocalCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, beta_coefficient):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.beta = beta_coefficient
        print(f"beta: {self.beta}")

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--beta-coefficient', default=1.0, type=float,
                            help='beta for the regularization parameter')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, orig_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'orig_loss': orig_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_local_consistency(self, orig_loss):
        shifted_loss = torch.roll(orig_loss, 1)
        difference = torch.add(shifted_loss, -1 * orig_loss)
        squared_difference = torch.mul(difference, difference)
        return squared_difference

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)

        lprobs_view = lprobs.view(-1, lprobs.size(-1))
        target_view = target.view(-1)

        orig_loss = F.nll_loss(
            lprobs_view,
            target_view,
            ignore_index=self.padding_idx,
            reduction='none',
        )

        local_consistency = self.compute_local_consistency(orig_loss)
        loss = torch.sum(orig_loss) + self.beta * torch.sum(local_consistency)
        
        return loss, torch.sum(orig_loss)  # also return the original loss (i.e., unregularized loss)

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('orig_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
        else:
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
