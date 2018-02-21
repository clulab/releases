from __future__ import division
from sklearn.metrics import r2_score
import numpy as np
import operator

def adj_r2(adjective, mean, adj_gold, adj_pred):
    num_data = len(adj_gold)
    assert len(adj_gold) == len(adj_pred)
    # print "Finding R2 for adjective:", adjective
    # Numerator
    squared_resids = [(adj_gold[i] - adj_pred[i])**2 for i in range(num_data)]
    sum_sq_resids = sum(squared_resids)
    # print "sum_sq_resids:", sum_sq_resids
    # Denominator
    squared_denoms = [(adj_gold[i] - mean)**2 for i in range(num_data)]
    sum_sq_denom = sum(squared_denoms)
    # print "sum_sq_denom:", sum_sq_denom
    # R2:
    r2 = 1 - (sum_sq_resids / sum_sq_denom)
    # print "\tR2 = ", r2
    return r2

def adj_mean_sq_err(adjective, adj_gold, adj_pred):
    num_data = len(adj_gold)
    assert len(adj_gold) == len(adj_pred)
    # print "Finding average squared error for adjective:", adjective
    # Numerator
    squared_resids = [(adj_gold[i] - adj_pred[i])**2 for i in range(num_data)]
    sum_sq_resids = sum(squared_resids)
    mean_squared_error = sum_sq_resids/num_data
    # print "\tMSE = ", avg_sum_squared_error
    return mean_squared_error, np.var(adj_gold)


# unseen data
fn = "data/NN_predictions_unseen.txt"
# seen data
# fn = "data/NN_predictions_seen.txt"
print("Opening file: " + fn)
lines = open(fn, 'r').readlines()
header = lines[0]
data = lines[1:]

all_gold = []
all_pred = []
all_adj = []

adj_dict = dict()

for line in data:
    fields = line.strip().split("\t")
    print("Fields: ", fields)
    assert len(fields) == 3

    gold = float(fields[0].strip())
    pred = float(fields[1].replace("[", "").replace("]", "").strip())
    adj = fields[2].strip()
    print("\t({0})\t({1})\t({2})".format(gold, pred, adj))
    all_gold.append(gold)
    all_pred.append(pred)
    all_adj.append(adj)
    if adj in adj_dict:
        adj_dict[adj]["gold"].append(gold)
        adj_dict[adj]["pred"].append(pred)
    else:
        adj_dict[adj] = {"gold": [], "pred": []}


print("Len{adj_dict) = ", len(adj_dict))

r2_all = r2_score(all_gold, all_pred)
print("\n")
print("R2 for everything: ", r2_all)
print("\n")

mean_overall = sum(all_gold)/len(all_gold)
# print "mean:", mean_overall
# print "variance:", np.var(all_gold)

# Find the MSE for each Adj
mse_by_adj = []
for curr_adj in adj_dict:
    curr_adj_r2 = adj_r2(curr_adj, mean_overall, adj_dict[curr_adj]["gold"], adj_dict[curr_adj]["pred"])
    curr_adj_mse, curr_adj_var = adj_mean_sq_err(curr_adj, adj_dict[curr_adj]["gold"], adj_dict[curr_adj]["pred"])
    mse_by_adj.append((curr_adj, curr_adj_mse, curr_adj_var, curr_adj_r2))

# Only did for unseen, but can do for seen as well...
mse_csv = open("/Users/bsharp/unseen_mse.csv", "w")
mse_csv.write(",".join(["adjective", "mse", "variance, r2"]) + "\n")
print "\tadjective\tmse\tvariance\tr2"
sorted_mse_by_adj = sorted(mse_by_adj, key=operator.itemgetter(3), reverse=True)
for a, mse, v, r2 in sorted_mse_by_adj:
    print "\t", a, "\t", round(mse,3), "\t", round(v, 3), "\t", round(r2, 3)
    mse_csv.write(",".join([str(x) for x in [a, mse, v, r2]]) + "\n")
mse_csv.close()



