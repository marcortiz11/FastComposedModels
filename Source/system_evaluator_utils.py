import numpy as np
import Source.io_util as io
import Source.protobuf.make_util as make
import Source.protobuf.FastComposedModels_pb2 as fcm
from enum import Enum
from time import perf_counter


class Metrics(object):

    __slots__ = ('accuracy',
                 'time',
                 'time_max',
                 'time_min',
                 'time_std',
                 'instances',  # How many instances executed by the model
                 'instance_model',  # Which model executes the instance?
                 'cm',  # Confusion matrix
                 'params',
                 'ops',)

    def __init__(self):
        self.time = 0
        self.accuracy = 0
        self.params = 0
        self.ops = 0
        self.time_max = 0
        self.time_min = 0
        self.time_std = 0
        self.instances = 0


class Results(object):
    __slots__ = ('train', 'test', 'val')


def get_accuracy_dictionary(pred, gt):
    num_inst = len(pred.keys())
    correct = [gt[key] == pred[key] for key in gt.keys()]
    acc = sum(correct) / num_inst
    return acc


def get_Lgtid(c_dict, phase, input_ids=None):
    """
    :param c_dict: Classifier dictionary
    :param phase: train or test
    :param input_ids: instances to be predicted
    :return: Logits, Ground Truth, IDs inputs
    """
    assert phase in c_dict, "ERROR: Classifier %s has no %s partition" % (c_dict['name'], phase)
    phase_dict = c_dict[phase]
    L = phase_dict['logits']
    gt = phase_dict['gt']
    id = phase_dict['id']
    if input_ids is not None:
        indices = np.where(np.isin(id, input_ids))
        indices = indices[0] if len(indices) > 0 else []
        return L[indices], gt[indices], id[indices]
    return L, gt, id


import importlib
import os


def train_trigger(sys, c):
    """
    Trains a trigger and creates its classifier whose name will be classifier's id and will be saved in
    Definitions/Classifiers/tmp
    :param sys: System of components
    :param c: Trigger protobuf definition
    :return: -
    """

    assert c.HasField("classifier"), "ERROR in TRIGGER.CLASSIFIER: A classifier should be specified for the trigger"
    train_trigger = not c.classifier.HasField("classifier_file") or c.classifier.classifier_file == ""
    if train_trigger:
        assert c.HasField("model"), \
            "ERROR in TRIGGER: A model must be specified when training the trigger"
        assert c.classifier.HasField("data_id"), \
            "ERROR in TRIGGER.CLASSIFIER: Training data should be specified when training the trigger"

        module = importlib.import_module('Definitions.triggers.'+c.model)
        c_dict = module.train_fit(sys, c.id)
        if 'TMP' in os.environ:
            tmp_location = os.path.join(os.environ['FCM'], os.environ['TMP']+'/')
        else:
            tmp_location = os.path.join(os.environ['FCM'], 'Definitions/Classifiers/tmp/')
        classifier_file = c.id
        io.save_pickle(tmp_location + classifier_file, c_dict)
        c.classifier.classifier_file = tmp_location + classifier_file
        sys.replace(c.id, c)


def pretty_print(results):
    print("TRAIN: ")
    for key in results.train:
        print("\t{}:".format(key))
        print("\t\t {} Accuracy".format(results.train[key].accuracy))
        print("\t\t {} # Ops".format(results.train[key].ops))
        print("\t\t {} # Params".format(results.train[key].params))
        print("\t\t Inference time:")
        print("\t\t\t {}sec Average".format(results.train[key].time))
        print("\t\t\t {}sec Max".format(results.train[key].time_max))
        print("\t\t\t {}sec Min".format(results.train[key].time_min))
    print("TEST: ")
    for key in results.test:
        print("\t{}:".format(key))
        print("\t\t {} Accuracy".format(results.test[key].accuracy))
        print("\t\t {} # Ops".format(results.test[key].ops))
        print("\t\t {} # Params".format(results.test[key].params))
        print("\t\t Inference time:")
        print("\t\t\t {}sec Average".format(results.test[key].time))
        print("\t\t\t {}sec Max".format(results.test[key].time_max))
        print("\t\t\t {}sec Min".format(results.test[key].time_min))
    print("VALIDATION: ")
    for key in results.val:
        print("\t{}:".format(key))
        print("\t\t {} Accuracy".format(results.val[key].accuracy))
        print("\t\t {} # Ops".format(results.val[key].ops))
        print("\t\t {} # Params".format(results.val[key].params))
        print("\t\t Inference time:")
        print("\t\t\t {}sec Average".format(results.val[key].time))
        print("\t\t\t {}sec Max".format(results.val[key].time_max))
        print("\t\t\t {}sec Min".format(results.val[key].time_min))


class CatchTime(object):

    def __enter__(self):
        self.time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.time