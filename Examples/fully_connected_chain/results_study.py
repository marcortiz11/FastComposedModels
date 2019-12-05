import Examples.paretto_front as front
import Source.io_util as io
import Examples.plot as myplt
import matplotlib.pyplot as plt
import numpy as np
import os


def results_together(folder):
    R = {}
    R_files = [r for r in os.listdir(folder) if "models" not in r]
    for r_file in R_files[::]:
        r = io.read_pickle(os.path.join(folder, r_file))
        R.update(r)
    return R


def get_speedup_front(f0,f1,e):
    """
    SUMMARY :Given a resource, compute the speedup of one front vs the other.
            Speedup = same accuracy less time
    :param f0: Front of classifiers/models 1
    :param f1: Front of classifiers/models 2
    :param e: Epsilon = ammount of resource (time,params)
    :return: Speedup of the front1 vs front0 given e
    """
    acc = 0
    time = 0
    for k, v in f0:
        if v['system'].time < e:
            acc = v['system'].accuracy
            time = v['system'].time

    speedup = 0
    for k, v in f1:
        if v['system'].time < e and v['system'].accuracy >= acc:
            speedup = time/v['system'].time
            break
    return speedup


def get_speedup_front(f0, f1, e, phase = None):
    """
    SUMMARY :Given a resource, compute the speedup of one front vs the other.
            Speedup = same accuracy less time
    :param f0: Front of classifiers/models 1
    :param f1: Front of classifiers/models 2
    :param e: Epsilon = ammount of resource (time,params)
    :return: Speedup of the front1 vs front0 given e
    """
    if phase == "test":
        f0 = [(k,f.test) for k,f in f0]
        f1 = [(k,f.test) for k,f in f1]

    acc = 0
    time = 0
    for k, v in f0:
        if v['system'].time < e:
            acc = v['system'].accuracy
            time = v['system'].time

    speedup = 0
    for k, v in f1:
        if v['system'].time < e:
            speedup = time/v['system'].time
        if v['system'].accuracy > acc: break
    return speedup


def get_increment_front(f0, f1, e, phase=None):
    """
    SUMMARY :Given a resource, compute the increment of one front vs the other.
            Speedup = more accuracy less time
    :param f0: Front of classifiers/models 1
    :param f1: Front of classifiers/models 2
    :param e: Epsilon = ammount of resource (time,params)
    :return: Speedup of the front1 vs front0 given e
    """

    if phase == "test":
        f0 = [(k,f.test) for k,f in f0]
        f1 = [(k,f.test) for k,f in f1]

    acc = 0
    for k, v in f0:
        if v['system'].time < e:
            acc = v['system'].accuracy

    increment = -1
    for k, v in f1:
        i = v['system'].accuracy - acc
        if v['system'].time < e and i > increment:
            increment = i
    return increment


if __name__ == "__main__":

    show = "all_solutions"
    pid = 103928

    # All models
    # fc = results_together("./results/front45_models_validation/"+str(pid))
    chain = results_together("./results/front45_models_validation/"+str(pid))
    models = io.read_pickle("./results/front45_models_validation/"+str(pid)+"/models.pkl")

    # Front
    # front_fc = front.get_front_time_accuracy(fc)
    front_chain = front.get_front_time_accuracy(chain)
    front_models = front.get_front_time_accuracy(models)

    # Sorted Front models
    # sorted_front_fc = front.sort_results_by_accuracy(front_fc)
    sorted_front_chain = front.sort_results_by_accuracy(front_chain)
    sorted_front_models = front.sort_results_by_accuracy(front_models)

    if show == "all_solutions":
        # myplt.plot_accuracy_time(fc, system_color='blue')
        myplt.plot_accuracy_time(chain)
        myplt.plot_accuracy_time(models)
        myplt.show()
    elif show == "fronts":
        # myplt.plot_accuracy_time(front_fc, system_color='blue')
        myplt.plot_accuracy_time(front_chain)
        myplt.plot_accuracy_time(front_models)
        myplt.show()
    elif show == "statistics_time_front":
        # Speedup
        x = []
        y = []
        for e in np.arange(10, 0, -0.2):
            speedup = get_speedup_front(sorted_front_models, sorted_front_chain, e)
            x.append(e)
            y.append(speedup)
            print("Speedup chain-models e=%f: %f" %(e, speedup))
        plt.plot(x, y, label="Chain speedup")
        print()
        x = []
        y = []
        for e in np.arange(10, 0, -0.1):
            speedup = get_speedup_front(sorted_front_models, sorted_front_fc, e)
            x.append(e)
            y.append(speedup)
            print("Speedup fc-models e=%f: %f" %(e, get_speedup_front(sorted_front_models, sorted_front_fc, e)))
        plt.plot(x, y, label="FC speedup")
        plt.legend()
        plt.show()
        print()

        # Increment in accuracy
        x = []
        y = []
        plt.yscale("log")
        for e in np.arange(10, 0, -0.5):
            increment = get_increment_front(sorted_front_models, sorted_front_chain, e)
            x.append(e)
            y.append(increment)
            print("Increment chain-models e=%f: %f" %(e, increment))
        plt.plot(x, y, label="Chain increment")
        print()
        x = []
        y = []
        for e in np.arange(10, 0, -0.5):
            increment = get_increment_front(sorted_front_models, sorted_front_fc, e)
            x.append(e)
            y.append(increment)
            print("Increment fc-models e=%f: %f" % (e, get_increment_front(sorted_front_models, sorted_front_fc, e)))
        plt.plot(x, y, label="FC increment")
        plt.legend()
        plt.show()
        print()
        # Increase in accuracy


