import numpy as np
import pickle
import time

# My intention is to put those functions in the io_utils.py later on
def save_pickle(filename, obj, format=".pkl", verbose=False):
    start = time.time()
    with open(filename+format, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        print("Saved pickle to: ", filename)
        print("Time to save:", time.time()-start)


def read_pickle(filename, format=".pkl", verbose=False):
    start = time.time()
    with open(filename+format, 'rb') as handle:
        classifier = pickle.load(handle)
    if verbose:
        print("Loaded pickle: ", filename)
        print("Time to load:", time.time() - start)
    return classifier


# My intention is to put those functions in make_utils.py later on
def make_classifier_raw_data(L, gt, ids):
    raw_data = {
        'logits': np.array(L),
        'gt': np.array(gt),
        'id': np.array(ids),
    }
    return raw_data


def make_performance_metrics(**args):
    performance = {  # Maybe we extend the metrics ?
        'time': args['time'] if 'time' in args else 0,
        'ops': args['ops'] if 'ops' in args else 0,
        'params': args['params'] if 'params' in args else 0,
    }
    return performance


def make_classifier(name, data_id, train_data, test_data, performance):
    classifier = {
        'name': name,
        'data_id': data_id,
        'train': train_data,
        'test': test_data,
        'metrics': performance,
    }
    return classifier


def check_valid_classifier(classifier):

    # TRAIN
    train_data = classifier['train']
    assert train_data['logits'].shape[0] == train_data['gt'].shape[0]\
            and train_data['logits'].shape[0] == len(set(train_data['id'])), \
        "ERROR in Classifier: number of instances should be consistent"
    assert train_data['logits'].shape[1] == len(set(train_data['gt'])), \
        "ERROR in Classifier: number of classses should be consistent"

    # TEST
    test_data = classifier['test']
    assert test_data['logits'].shape[0] == test_data['gt'].shape[0] \
           and test_data['logits'].shape[0] == len(set(test_data['id'])), \
        "ERROR in Classifier: number of instances should be consistent"
    assert test_data['logits'].shape[1] == len(set(test_data['gt'])), \
        "ERROR in Classifier: number of classses should be consistent"


if __name__ == '__main__':

    # CREATE and SAVE!
    n_instances = int(1*1e4)
    n = 10

    Logits = np.random.rand(n_instances, n)
    gt = np.random.randint(0, n, n_instances)
    ids = np.arange(n_instances)  # Considering making ids optional

    train = make_classifier_raw_data(Logits, gt, ids)
    test = make_classifier_raw_data(Logits, gt, ids)
    performance = make_performance_metrics(**{'time': 1e-3, 'ops': 1e15, 'params': 1e8})
    classifier = make_classifier("V001_ResNet20_CIFAR10", train, test, performance)

    save_pickle("V001_ResNet20_CIFAR10", classifier, verbose=True)

    # READ!
    classifier = read_pickle("V001_ResNet20_CIFAR10", verbose=True)
    check_valid_classifier(classifier)
    # print(classifier)
    print("Test accuracy:", sum(np.argmax(classifier['test']['logits'], axis=1) == classifier['test']['gt'])/n_instances)