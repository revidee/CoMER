from comer.utils.utils import Hypothesis


def score_avg(hyp: Hypothesis):
    return hyp.score / 2

def th_fn_avg(hyp, th):
    seq_len = len(hyp.seq)
    return (score_avg(hyp) >= th) if seq_len else False


def score_min(hyp: Hypothesis):
    return min(hyp.history)


def th_fn_min(hyp: Hypothesis, th):
    seq_len = len(hyp.seq)
    return (score_min(hyp) >= th) if seq_len else False


def score_bimin(hyp: Hypothesis):
    return min((min(hyp.history), min(hyp.best_rev)))


def th_fn_bimin(hyp: Hypothesis, th):
    seq_len = len(hyp.seq)
    return (score_bimin(hyp) >= th) if seq_len else False


def score_sum(hyp: Hypothesis):
    return sum(hyp.history)


def th_fn_sum(hyp, th):
    seq_len = len(hyp.seq)
    return (score_sum(hyp) >= th) if seq_len else False

def score_bisum(hyp: Hypothesis):
    return sum(hyp.history) + sum(hyp.best_rev)

def th_fn_bisum(hyp, th):
    seq_len = len(hyp.seq)
    return (score_bisum(hyp) >= th) if seq_len else False

def score_bisum_avg(hyp: Hypothesis):
    return (sum(hyp.history) + sum(hyp.best_rev)) / 2

def th_fn_bisum_avg(hyp, th):
    seq_len = len(hyp.seq)
    return (score_bisum_avg(hyp) >= th) if seq_len else False
