import numpy as np

def get_map(pred_scores, target_indices):
    # pred_scores: the raw score of the predictions given by the linear probe.
    # target_indices: raw indices of the tokens that we want to predict
    assert(isinstance(pred_scores, np.ndarray))
    assert(isinstance(target_indices, np.ndarray))

    pred_rankings = np.zeros(len(pred_scores))
    pred_rankings[np.flip(pred_scores.argsort())] = np.arange(len(pred_scores))+1
    rankings_targets = pred_rankings[target_indices]

    rankings_targets_sorted = np.sort(rankings_targets)
    numerator_sorted = (np.arange(len(target_indices))+1)
    map = np.mean(numerator_sorted/rankings_targets_sorted)

    return map

def get_ppl_(pred_scores, target_indices):
    assert (isinstance(pred_scores, np.ndarray))
    assert (isinstance(target_indices, np.ndarray))

    n_target_token = len(target_indices)

    probs_raw = pred_scores[target_indices]

    ppl = np.power(np.prod(1/probs_raw),1/n_target_token)

    return ppl

def get_ppl(pred_scores, target_indices):
    assert (isinstance(pred_scores, np.ndarray))
    assert (isinstance(target_indices, np.ndarray))

    n_target_token = len(target_indices)

    probs_raw = pred_scores[target_indices]

    ppl = np.exp(np.sum(np.log(1/probs_raw))/n_target_token)

    return ppl
