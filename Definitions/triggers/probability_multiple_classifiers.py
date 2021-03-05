import Source.protobuf.make_util as make
import Source.io_util as io
from Source.math_util import softmax
import numpy as np

def __ops():
    return 0


def __parameters():
    return 1


def __time():
    return 0


def __get_trigger_raw_data(data, phase):

    if phase == "test":
        data_source = data.source.test_path
    elif phase == "train":
        data_source = data.source.train_path
    else:  # phase == "val":
        data_source = data.source.val_path

    dataset = io.read_pickle(data_source)

    L = dataset['logits']
    P = softmax(L)
    ths = dataset['ths']
    gt = dataset['gt']
    ids = dataset['id']

    sort = np.sort(P, axis=1)
    diff = sort[:, -1] - sort[:, -2]

    sample_size = len(ids)
    Y = np.zeros((sample_size, len(ths)+1))  # Predictions for the trigger

    Or = np.zeros(sample_size)
    for ith, th in enumerate(ths):
        Y[:, ith] = np.array([diff < th])
        Or = np.logical_or(Or, Y[:, ith])

    Y[:, -1] = np.logical_not(Or)  # Executed by classifier before trigger if none matches

    raw_data = make.make_classifier_raw_data(Y, gt == np.argmax(L, axis=1), ids)
    return raw_data


def train_fit(sys, id_trigger):

    trigger = sys.get(id_trigger)
    data_id = trigger.classifier.data_id
    data = sys.get(data_id)

    # Test
    test = __get_trigger_raw_data(data, "test")
    # Train
    train = __get_trigger_raw_data(data, "train")
    # Validation
    val = None
    if data.source.HasField('val_path'):
        val = __get_trigger_raw_data(data, "val")

    # obtain metrics
    metrics = make.make_performance_metrics(**{'time': __time(),
                                               'ops': __ops(),
                                               'params': __parameters()
                                               })
    # Create dict
    classifier_trigger_dict = make.make_classifier_dict("trigger_classifier", data_id,
                                                        train, test, metrics, val_data=val)
    return classifier_trigger_dict
