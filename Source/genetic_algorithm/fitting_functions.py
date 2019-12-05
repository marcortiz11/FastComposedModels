import Source.system_evaluator as eval
import numpy as np


def f1_time_penalization(P, a=20, b=1, time_constraint = None):
    """
    F() = a*acc + b*(t/tr)
    :param a: Weights accuracy
    :param b: Weights time
    :param c: Weights parameters
    :return:
    """
    fit = np.zeros(len(P))
    for i_ind, i in enumerate(P):
        start = i.get_start()
        result = eval.evaluate(i, start, phases=["val"])
        acc = result.val['system'].accuracy
        time = result.val['system'].time
        decrease_time = (time_constraint-time)
        fit[i_ind] = a*acc + b*decrease_time
    return fit

