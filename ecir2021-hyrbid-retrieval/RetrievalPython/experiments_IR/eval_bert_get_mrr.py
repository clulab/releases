import numpy as np


openbook_mrr = {}
openbook_mrr["test"] = np.array([0.51853131, 0.59501887, 0.56521779, 0.55736404, 0.55368897])


squad_mrr = {}
squad_mrr["test"] = np.array([0.26856396, 0.26955579, 0.22217043, 0.26713424, 0.2744626])

print("openbook test mean:", np.mean(openbook_mrr["test"]))
print("squad test mean:", np.mean(squad_mrr["test"]))