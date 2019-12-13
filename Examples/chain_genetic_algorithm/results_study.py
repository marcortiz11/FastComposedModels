import Examples.paretto_front as front
import Source.io_util as io
import Examples.plot as myplt
import matplotlib.pyplot as plt
import numpy as np
import os


def get_speedup_front(f0, f1, t, phase = None):
    """
    SUMMARY :Given a resource, compute the speedup of one front vs the other.
            Speedup = same accuracy less time
    :param f0: Front of classifiers/models 1
    :param f1: Front of classifiers/models 2
    :param e: Epsilon = ammount of resource (time,params)
    :return: Speedup of the front1 vs front0 given e
    """
    if phase == "test":
        f0 = [(k, f.test) for k, f in f0]
        f1 = [(k, f.test) for k, f in f1]

    acc = 0
    time = 0
    for k, v in f0:
        if v['system'].time < t:
            acc = v['system'].accuracy
            time = v['system'].time

    time_ensemble = -1
    for k, v in f1:
        if v['system'].accuracy >= acc:
            time_ensemble = v['system'].time
            break
        elif v['system'].time > time:
            break
        time_ensemble = v['system'].time

    return time/time_ensemble


def get_increment_front(f0, f1, t, phase=None):
    """
    SUMMARY :Given a resource, compute the increment of one front vs the other.
            Speedup = more accuracy less time
    :param f0: Front of classifiers/models 1
    :param f1: Front of classifiers/models 2
    :param e: Epsilon = ammount of resource (time,params)
    :return: Speedup of the front1 vs front0 given e
    """

    if phase == "test":
        f0 = [(k, f.test) for k, f in f0]
        f1 = [(k, f.test) for k, f in f1]

    acc = 0
    for k, v in f0:
        if v['system'].time < t:
            acc = v['system'].accuracy

    increment = -1
    for k, v in f1:
        i = v['system'].accuracy - acc
        if v['system'].time < t and i > increment:
            increment = i

    return increment


if __name__ == "__main__":

    show = "statistics_time_front"
    pid = 103457
    time_constraints = [2, 4, 6]

    # All models
    # fc = results_together("./results/front45_models_validation/"+str(pid))
    chain = io.read_pickle("./results/front45_models_validation/"+str(pid)+'/R.pkl')
    models = io.read_pickle("./results/front45_models_validation/models.pkl")

    # Front
    front_chain = front.get_front_time_accuracy(chain, phase="test")
    front_models = front.get_front_time_accuracy(models, phase="test")

    # Sorted Front mod  els
    sorted_front_chain = front.sort_results_by_accuracy(front_chain, phase="test")
    sorted_front_models = front.sort_results_by_accuracy(front_models, phase="test")

    if show == "all_solutions":
        myplt.plot_accuracy_time_old(chain)
        myplt.plot_accuracy_time_old(models)
        myplt.show()
    elif show == "fronts":
        myplt.plot_accuracy_time_old(front_chain)
        myplt.plot_accuracy_time_old(front_models)
        myplt.show()
    elif show == "statistics_time_front":

        # Speedup
        for e in time_constraints:
            speedup = get_speedup_front(sorted_front_models, sorted_front_chain, e, phase="test")
            print("Speedup chain-models T=%f: %f" %(e, speedup))

        # Increment in accuracy
        plt.yscale("log")
        for e in time_constraints:
            increment = get_increment_front(sorted_front_models, sorted_front_chain, e, phase="test")
            print("Increment chain-models T=%f: %f" %(e, increment))


