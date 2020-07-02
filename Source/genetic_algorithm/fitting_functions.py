import Source.system_evaluator as eval


def f1_time_param_penalization(P_r, w=[0.33, 0.33, 0.33], phase="val"):

    """
    :param P_r: Population evaluation results R
    :param w: Weights for accuracy and time objectives
    :param phase: Compute fitness based on train, test or validation splits
    :return: fitness values for each individual
    """

    assert len(w) == 3, "Lenth of w should be = 3"

    fit = [0] * len(P_r)
    max_time = max([i.val['system'].time if phase == 'val' else i.test['system'].time for i in P_r])
    max_params = max([i.val['system'].params if phase == 'val' else i.test['system'].params for i in P_r])

    for i_ind, i in enumerate(P_r):

        if phase == 'val':
            acc = i.val['system'].accuracy
            inference_time = 1 - i.val['system'].time/max_time
            model_params = 1 - i.val['system'].params/max_params
        else:
            acc = i.test['system'].accuracy
            inference_time = 1 - i.test['system'].time / max_time
            model_params = 1 - i.test['system'].params / max_params

        fit[i_ind] = pow((w[0] * acc) + (w[1] * inference_time) + (w[2] * model_params), 3)
    return fit


def f1_time_penalization(P_r, w=[0.5, 0.5], phase="val"):

    """
    :param P_r: Population evaluation results R
    :param w: Weights for accuracy and time objectives
    :param phase: Compute fitness based on train, test or validation splits
    :return: fitness values for each individual
    """

    assert len(w) == 3, "Lenth of w should be = 2"

    fit = [0] * len(P_r)
    max_time = max([i.val['system'].time if phase == 'val' else i.test['system'].time for i in P_r])

    for i_ind, i in enumerate(P_r):

        if phase == 'val':
            acc = i.val['system'].accuracy
            inference_time = 1 - i.val['system'].time/max_time
        else:
            acc = i.test['system'].accuracy
            inference_time = 1 - i.test['system'].time / max_time

        fit[i_ind] = pow((w[0] * acc) + (w[1] * inference_time), 3)
    return fit


def f1(P_r, w=[1], phase="val"):

    """
    :param P_r: Population evaluation results R
    :param w: Weights for accuracy and time objectives
    :param phase: Compute fitness based on train, test or validation splits
    :return: fitness values for each individual
    """
    assert len(w) == 1, "Lenght of w should be 1"

    fit = [0] * len(P_r)
    for i_ind, i in enumerate(P_r):
        if phase == 'val':
            acc = i.val['system'].accuracy
        else:
            acc = i.test['system'].accuracy
        fit[i_ind] = pow((w[0]*acc), 3)

    return fit