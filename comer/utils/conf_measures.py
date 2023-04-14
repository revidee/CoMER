from comer.utils.utils import Hypothesis


def score_ori(hyp: Hypothesis):
    return hyp.score / 2

def th_fn_ori(hyp, th):
    seq_len = len(hyp.seq)
    return (score_ori(hyp) >= th) if seq_len else False


def score_min(hyp: Hypothesis):
    seq_len = len(hyp.seq)
    return min(hyp.history) if seq_len else float('-Inf')


def th_fn_min(hyp: Hypothesis, th):
    seq_len = len(hyp.seq)
    return (score_min(hyp) >= th) if seq_len else False


def score_bimin(hyp: Hypothesis):
    seq_len = len(hyp.seq)
    return min((min(hyp.history), min(hyp.best_rev))) if seq_len else float('-Inf')


def th_fn_bimin(hyp: Hypothesis, th):
    seq_len = len(hyp.seq)
    return (score_bimin(hyp) >= th) if seq_len else False


def score_sum(hyp: Hypothesis):
    return sum(hyp.history)

def th_fn_sum(hyp, th):
    seq_len = len(hyp.seq)
    return (score_sum(hyp) >= th) if seq_len else False

def score_avg(hyp: Hypothesis):
    seq_len = len(hyp.seq)
    return sum(hyp.history) / seq_len if seq_len else float('-Inf')

def th_fn_avg(hyp, th):
    seq_len = len(hyp.seq)
    return (score_avg(hyp) >= th) if seq_len else False

def score_rev_sum(hyp: Hypothesis):
    return sum(hyp.best_rev)

def score_rev_avg(hyp: Hypothesis):
    seq_len = len(hyp.seq)
    return sum(hyp.best_rev) / seq_len if seq_len else float('-Inf')

def score_bisum(hyp: Hypothesis):
    return sum(hyp.history) + sum(hyp.best_rev)

def th_fn_bisum(hyp, th):
    seq_len = len(hyp.seq)
    return (score_bisum(hyp) >= th) if seq_len else False

def score_bisum_avg(hyp: Hypothesis):
    return (sum(hyp.history) + sum(hyp.best_rev)) / 2

def score_bi_avg(hyp: Hypothesis):
    seq_len = len(hyp.seq)
    return (sum(hyp.history) + sum(hyp.best_rev)) / (2 * seq_len) if seq_len else float('-Inf')

def th_fn_bisum_avg(hyp, th):
    seq_len = len(hyp.seq)
    return (score_bisum_avg(hyp) >= th) if seq_len else False


CONF_MEASURES = {
    'ori': score_ori,
    'min': score_min,
    'bimin': score_bimin,
    'mult': score_sum,
    'revmult': score_rev_sum,
    'bimult': score_bisum,
    'bimultavg': score_bisum_avg,
    'avg': score_avg,
    'biavg': score_bi_avg
}
