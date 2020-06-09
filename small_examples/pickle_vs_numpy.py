import numpy as np
import pickle
import Source.io_util as io
import Source.make_util as make
import time

n_entries = int(1e6)
n = 10

logits = np.random.rand(n_entries, n)
gt = np.random.randint(n_entries)
id_entries = np.arange(n_entries)

id = "dummy_classifier"
flops = 1e10
time_inf = 1e-2
params = 1e9

classifier = {
    'logits': logits,
    'gt': gt,
    'id_entries': id_entries
}

start = time.time()
with open('classifier.pickle', 'wb') as handle:
    pickle.dump(classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Pickle write time:", time.time() - start)

start = time.time()
with open('classifier.pickle', 'rb') as handle:
    classifier = pickle.load(handle)

print("Pickle read time:", time.time() - start)
