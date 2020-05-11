import time
import pickle


def save_message(file_name, msg, suffix=".prototxt", verbose=True):
    start = time.time()
    f = open(file_name + suffix, "wb")
    f.write(msg.SerializeToString())
    f.close()
    if verbose:
        print("File {} saved.".format(file_name + suffix))
        print("Time to save:", time.time() - start)


def read_message(file_name, msg, suffix=".prototxt", verbose=True):
    start = time.time()
    f = open(file_name + suffix, "rb")
    msg.ParseFromString(f.read())
    f.close()
    if verbose:
        print("File {} loaded.".format(file_name + suffix))
        print("Time to load:", time.time() - start)


def save_pickle(file_name, obj, suffix=".pkl", verbose=False):
    start = time.time()
    file_name = file_name + suffix if suffix not in file_name else file_name
    with open(file_name, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        print("File {} saved.".format(file_name))
        print("Time to s    ave:", time.time() - start)


def read_pickle(file_name, suffix=".pkl", verbose=False):
    start = time.time()
    file_name = file_name+suffix if suffix not in file_name else file_name
    with open(file_name, 'rb') as handle:
        classifier = pickle.load(handle)
    if verbose:
        print("File {} loaded.".format(file_name))
        print("Time to load:", time.time() - start)
    return classifier
