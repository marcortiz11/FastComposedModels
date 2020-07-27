
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
    assert phase == "val" or phase == "test" or phase == "train", "ERROR: Fitness computed on test or validation splits"

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

        fit[i_ind] = pow((w[0] * relative_acc) + (w[1] * relative_inference_time) + (w[2] * relative_model_params), 3)

    return fit

