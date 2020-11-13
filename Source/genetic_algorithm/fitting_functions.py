import numpy as np

"""
class Transform:

    def set_max_features(self, max):
        assert len(max) > 0, "Should be a list"
        self.max = max

    def set_min_features(self, min):
        assert len(min) > 0, "Should be a list"
        self.min = min

    def scale(self, *features):
        assert len(features) == len(self.max)

        scaled = []
        for fi, f in enumerate(features):
            scaled.append(f/self.max[fi])
        return scaled

    def min_max_normalize(self, *features):
        assert len(features) == len(self.max) and len(features) == len(self.min)

        min_max_normalized = []
        for fi, f in enumerate(features):
            assert self.min < f < self.max, "Features in different order"
            min_max_normalized.append((f - self.min[fi]) / (self.max[fi] - self.min[fi]))
        return min_max_normalized
"""

def make_limits_dict():

    limits = {'max_accuracy': None,
              'min_accuracy': None,
              'max_time': None,
              'min_time': None,
              'max_params': None,
              'min_params': None,
              'max_ops': None,
              'min_ops': None
              }

    return limits


def update_limit_dict(limit, R, phase="val"):

    assert "train" == phase or "test" == phase or "val" == phase, "%s not in test, val or train" % phase

    if phase == "val":
        R_phase = [i.val for i in R.values()]
    elif phase == "test":
        R_phase = [i.test for i in R.values()]
    else:
        R_phase = [i.train for i in R.values()]

    max_time = max([i['system'].time for i in R_phase])
    min_time = min([i['system'].time for i in R_phase])
    max_params = max([i['system'].params for i in R_phase])
    min_params = min([i['system'].params for i in R_phase])
    max_accuracy = max([i['system'].accuracy for i in R_phase])
    min_accuracy = min([i['system'].accuracy for i in R_phase])

    if 'max_accuracy' in limit:
        limit['max_accuracy'] = max_accuracy
    if 'min_accuracy' in limit:
        limit['min_accuracy'] = min_accuracy
    if 'max_time' in limit:
        limit['max_time'] = max_time
    if 'min_time' in limit:
        limit['min_time'] = min_time
    if 'max_params' in limit:
        limit['max_params'] = max_params
    if 'min_params' in limit:
        limit['min_params'] = min_params


def f1_time_param_penalization(P_r, w, limits, phase="val"):

    """
    Computes a numerical fitness value regarding 3 objectives:
        Fit() = w1*norm(acc) + w2*norm(time) + w3*norm(params)
    The three: acc, time and params are scaled regarding values specified in limits

    :param P_r: Evaluated results of population P
    :param w: Weights rewarding/penalizing each objective
    :param limits: Scaling constants
    :param phase: Dataset split. Train/Test/Validation
    :return: A python list ==> fitness value for each individual
    """

    assert len(w) == 3, "Lenth of w should be = 3"
    assert phase == "val" or phase == "test", "ERROR: Fitness computed on test or validation splits"

    fit = [0] * len(P_r)
    max_time = max([i.val['system'].time if phase == 'val' else i.test['system'].time for i in P_r])
    max_params = max([i.val['system'].params if phase == 'val' else i.test['system'].params for i in P_r])
    max_acc = max([i.val['system'].accuracy if phase == 'val' else i.test['system'].accuracy for i in P_r])

    for i_ind, i in enumerate(P_r):

        if phase == 'val':
            acc = i.val['system'].accuracy/max_acc
            inference_time = 1 - i.val['system'].time/max_time
            model_params = 1 - i.val['system'].params/max_params
        else:
            acc = i.test['system'].accuracy/max_acc
            inference_time = 1 - i.test['system'].time / max_time
            model_params = 1 - i.test['system'].params / max_params

        fit[i_ind] = pow((w[0] * acc) + (w[1] * inference_time) + (w[2] * model_params), 3)
    return fit


def f2_time_param_penalization(P_r, w, limits, phase="val"):

    """
    Computes a numerical fitness value regarding 3 objectives:
        Fit() = w1*norm(acc) + w2*norm(time) + w3*norm(params)
    The three: acc, time and params are normalized with min-max normalization

    :param P_r: Evaluated results of population P
    :param w: Weights rewarding/penalizing each objective
    :param limits: min&max constants for normalizaing
    :param phase: Dataset split. Train/Test/Validation
    :return: A python list ==> fitness value for each individual
    """

    assert len(w) == 3, "Lenth of w should be = 3"
    assert phase == "val" or phase == "test" or phase == "train", "ERROR: Fitness computed on test, val or train splits"

    fit = [0] * len(P_r)

    for i_ind, i in enumerate(P_r):
        if phase == 'val':
            absolute_acc = i.val['system'].accuracy
            absolute_time = i.val['system'].time
            absolute_params = i.val['system'].params
        elif phase == 'test':
            absolute_acc = i.test['system'].accuracy
            absolute_time = i.test['system'].time
            absolute_params = i.test['system'].params
        else:
            absolute_acc = i.train['system'].accuracy
            absolute_time = i.train['system'].time
            absolute_params = i.train['system'].params

        relative_acc = (absolute_acc - limits['min_accuracy'])/(limits['max_accuracy'] - limits['min_accuracy'])
        relative_inference_time = (absolute_time - limits['max_time'])/(limits['min_time'] - limits['max_time'])
        relative_model_params = (absolute_params - limits['max_params'])/(limits['min_params'] - limits['max_params'])

        fit[i_ind] = (w[0] * relative_acc) + (w[1] * relative_inference_time) + (w[2] * relative_model_params)

    return fit


def normalize_error_time_params(P_r: list, limits: dict, phase="val") -> np.ndarray:

    assert phase == "val" or phase == "test" or phase == "train", "ERROR: Fitness computed on test, val or train splits"

    if phase == 'val':
        absolute_err = [1 - i.val['system'].accuracy for i in P_r]
        absolute_time = [i.val['system'].time for i in P_r]
        absolute_params = [i.val['system'].params for i in P_r]
    elif phase == 'test':
        absolute_err = [1 - i.test['system'].accuracy for i in P_r]
        absolute_time = [i.test['system'].time for i in P_r]
        absolute_params = [i.test['system'].params for i in P_r]
    else:
        absolute_err = [1 - i.train['system'].accuracy for i in P_r]
        absolute_time = [i.train['system'].time for i in P_r]
        absolute_params = [i.train['system'].params for i in P_r]

    relative_err = np.array([(abs_err - (1-limits['max_accuracy']))/
                            (1-limits['min_accuracy'] - (1-limits['max_accuracy'])) for abs_err in absolute_err])
    relative_time = np.array([(abs_time - limits['min_time'])/
                            (limits['max_time'] - limits['min_time']) for abs_time in absolute_time])
    relative_params = np.array([(abs_params - limits['min_params'])/
                                (limits['max_params'] - limits['min_params']) for abs_params in absolute_params])

    fitness = np.array([relative_err, relative_time, relative_params]).T
    return fitness
