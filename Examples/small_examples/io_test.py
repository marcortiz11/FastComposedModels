import numpy as np
import time
import pickle

def write001():
    n = int(1e6)
    num_c = 10
    logits = np.random.rand(n, num_c)

    st = time.time()
    np.save("logits.npy", logits)
    d = time.time() - st
    print("Save time: {:.3f} sec(s)".format(d))


if __name__ == '__main__':
    n = int(1e6)
    num_c = 10
    logits = np.random.rand(n, num_c)
    x = np.random.rand(n)
    v = [logits, x]

    st = time.time()
    pickle.dump(v, open("dummy.pkl", "wb"))
    d = time.time() - st

    print("Save time: {:.3f} sec(s)".format(d))
