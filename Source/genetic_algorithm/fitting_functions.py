import Source.system_evaluator as eval


def f1_time_penalization(P, a=100, b=1, time_constraint = None):
    """
    F(acc, time) = a*acc + b*(t/tr)
    :param a: Weights accuracy
    :param b: Weights time
    :return:
    :param c: Weights parameters
    """
    fit = [0]*len(P)
    import time
    for i_ind, i in enumerate(P):
        start = i.get_start()

        time_start = time.time()
        result = eval.evaluate(i, start, phases=["val"])
        print("\t eval time %f:" % (time.time()-time_start))

        # Measurements
        acc = result.val['system'].accuracy
        time_inference = result.val['system'].time
        decrease_time = (time_constraint-time_inference)
        fit[i_ind] = pow(a*acc + b*decrease_time, 3)

    return fit



def f1_time_penalization_preevaluated(P_r, a=5):
    """
    F(acc, time) = a*acc + b*(t/tr)
    :param P_r: List of evaluation results for each individual in the population
    :param a: Weights accuracy
    :param b: Weights time
    :param time_constraint: Time to not surpass
    :return: Fitting values
    """
    fit = [0]*len(P_r)
    for i_ind, i in enumerate(P_r):
        acc = i.val['system'].accuracy
        time_inference = i.val['system'].time
        fit[i_ind] = pow(a*acc - time_inference, 3)
    return fit


def f2_time_penalization_preevaluated(P_r, a=5, phase='val'):
    """
    F(acc, time) = a*acc + b*(t/tr)
    :param a: Weights accuracy
    :param b: Weights time
    :return:
    :param c: Weights parameters
    """
    fit = [0] * len(P_r)
    max_time = max([i.val['system'].time for i in P_r])
    for i_ind, i in enumerate(P_r):
        if phase == 'val':
            acc = i.val['system'].accuracy
            inference_time = i.val['system'].time/max_time
        else:
            acc = i.test['system'].accuracy
            inference_time = i.test['system'].time / max_time
        fit[i_ind] = pow((a * acc) / inference_time, 3)
    return fit


def f1_time_param_penalization(P_r, a=5, phase="val"):

    """
    F(acc, time) = a*test_accuracy / (inference_time*model_params)
    :param a: Weights accuracy
    :param b: Weights time
    :return:
    :param c: Weights parameters
    """

    fit = [0] * len(P_r)
    max_time = max([i.val['system'].time if phase == 'val' else i.test['system'].time for i in P_r])
    max_params = max([i.val['system'].params if phase == 'val' else i.test['system'].params for i in P_r])

    for i_ind, i in enumerate(P_r):

        if phase == 'val':
            acc = i.val['system'].accuracy
            inference_time = i.val['system'].time/max_time
            model_params = i.val['system'].params/max_params
        else:
            acc = i.test['system'].accuracy
            inference_time = i.test['system'].time / max_time
            model_params = i.test['system'].params / max_params

        fit[i_ind] = pow((a*acc) - inference_time - model_params, 3)
    return fit