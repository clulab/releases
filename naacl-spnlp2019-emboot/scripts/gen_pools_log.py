from collections import defaultdict

def read_results():
    with open('sgd_results_noisy_or.txt') as f:
        for line in f:
            [score, label, entity] = line.strip().split('\t')
            yield (label, entity)

results = defaultdict(list)

for (label, entity) in read_results():
    results[label].append(entity)

for i in range(21):
    print 'Epoch', i
    for label in results:
        entities = results[label][:10]
        results[label] = results[label][10:]
        chunks = [label] + entities
        print '\t'.join(chunks)
