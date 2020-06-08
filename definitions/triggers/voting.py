import source.make_util as make
import source.io_util as io
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
    else:
        data_source = data.source.val_path

    dataset = io.read_pickle(data_source)

    P = dataset['predictions']  # Models for voting protocol
    n = dataset['n']
    gt = dataset['gt']
    ids = dataset['ids']

    agree = np.apply_along_axis(np.bincount, 1, P, None, P.shape[2])
    agree_max = np.max(agree, axis=1)
    y = np.transpose(np.append(np.array([agree_max < n]), np.array([agree_max >= n]), axis=0))
    raw_data = make.make_classifier_raw_data(y, gt == np.argmax(P, axis=1), ids)
    return raw_data


def train_fit(sys, id_trigger):

    trigger = sys.get(id_trigger)
    data_id = trigger.classifier.data_id
    data = sys.get(data_id)

    # Test
    test = __get_trigger_raw_data(data, "test")
    # Train
    train = None #__get_trigger_raw_data(data, "train")
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
