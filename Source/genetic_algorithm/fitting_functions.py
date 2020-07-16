import Source.system_evaluator as eval


def f1_time_param_penalization(P_r, w=[0.33, 0.33, 0.33], phase="val"):

    """
    Fit() = w1*scale(acc) + w2*scale(time) + w3*scale(params)

    :param P_r: Population evaluation results R
    :param w: Weights for accuracy and time objectives
    :param phase: Compute fitness based on train, test or validation splits
    :return: fitness values for each individual
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


def f2_time_param_penalization(P_r, w=[0.33, 0.33, 0.33], phase="val"):

    """

    Fit() = w1*norm(acc) + w2*norm(time) + w3*norm(params)

    :param P_r: Population evaluation results R
    :param w: Weights for accuracy and time objectives
    :param phase: Compute fitness based on train, test or validation splits
    :return: fitness values for each individual
    """

    assert len(w) == 3, "Lenth of w should be = 3"
    assert phase == "val" or phase == "test", "ERROR: Fitness computed on test or validation splits"

    fit = [0] * len(P_r)
    max_time = max([i.val['system'].time if phase == 'val' else i.test['system'].time for i in P_r])
    min_time = min([i.val['system'].time if phase == 'val' else i.test['system'].time for i in P_r])
    max_params = max([i.val['system'].params if phase == 'val' else i.test['system'].params for i in P_r])
    min_params = min([i.val['system'].params if phase == 'val' else i.test['system'].params for i in P_r])
    max_acc = max([i.val['system'].accuracy if phase == 'val' else i.test['system'].accuracy for i in P_r])
    min_acc = min([i.val['system'].accuracy if phase == 'val' else i.test['system'].accuracy for i in P_r])

    for i_ind, i in enumerate(P_r):

        if phase == 'val':
            acc = (i.val['system'].accuracy - min_acc)/(max_acc - min_acc)
            inference_time = 1 - (i.val['system'].time - min_time)/(max_time - min_time)
            model_params = 1 - (i.val['system'].params - min_params)/(max_params - min_params)
        else:
            acc = (i.test['system'].accuracy - min_acc) / (max_acc - min_acc)
            inference_time = 1 - (i.test['system'].time - min_time) / (max_time - min_time)
            model_params = 1 - (i.test['system'].params - min_params) / (max_params - min_params)

        fit[i_ind] = pow((w[0] * acc) + (w[1] * inference_time) + (w[2] * model_params), 3)
    return fit

