# Author: Fan luo

def prediction_correctness(preds, gold):
    r = []
    gold = [g.lower().strip() for g in gold]
    for pred in preds:
        pred = pred.lower().strip()
        if pred in gold:
            r.append(1)
        else:
            r.append(0)
     
    return r


def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k]
    if r.size == 0:
        return 0
    if r.size != k:
        return np.sum(r) / k 
    else:
        return np.mean(r)


def recall_at_k(preds, gold, k):
    assert k >= 1
    assert len(gold) >= 1
    count = 0
    preds = preds[0:k]
    for a in gold:
        if a in preds:
            count += 1
    result = count / len(gold)
    return result

def average_precision(r, m):  # m: len(gold)
    assert m >= 1
    r = np.asarray(r)
    if r.size == 0:
        return 0 
    out = []
    for k in range(r.size): 
        if r[k]:  
            out.append(precision_at_k(r, k + 1))  # if r[k]: one of gold
    if not out:  # none of gold in predictions, out=[]
        return 0 
    return np.sum(out) / m
 
def retriveal_eval(sp, ranked_sents):  
 
    # scores
    r = prediction_correctness(ranked_sents, sp)
    p_2 = precision_at_k(r,2)
    p_3 = precision_at_k(r,3)
    p_5 = precision_at_k(r,5)
    p_10 =precision_at_k(r,10)
    p_20 =precision_at_k(r,20)
    m = len(sp)
    ap = average_precision(r, m) 
    
    recall_2 = recall_at_k(ranked_sents, sp, 2)
    recall_3 = recall_at_k(ranked_sents, sp, 3)
    recall_5 = recall_at_k(ranked_sents, sp, 5)
    recall_10 = recall_at_k(ranked_sents, sp, 10)
    recall_20 = recall_at_k(ranked_sents, sp, 20)
    
    res = {'p_2': p_2, 'p_3': p_3, 'p_5': p_5, 'p_10': p_10, 'p_20': p_20, 'ap': ap, 
           'recall_2': recall_2, 'recall_3': recall_3, 'recall_5': recall_5, 'recall_10': recall_10, 'recall_20': recall_20
          }   
    return res
    
def avg_score(scores, metric):
    score_sum = 0
    for score in scores:
        score_sum += score[metric]
    return round(score_sum / len(scores), 2)
