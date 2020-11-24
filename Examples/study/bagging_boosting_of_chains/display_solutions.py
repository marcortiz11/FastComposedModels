import Examples.metadata_manager_results as results_manager
from Examples.study.paretto_front import get_front_time_accuracy
from Source.genetic_algorithm.moo import non_dominated_selection
import Source.io_util as io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
import numpy as np
import os


def acc_time_subplot(Y, X, color, title="Solution space", label=None, print_front=False, alpha=0.2, s=5, marker="v"):
    splt = plt.subplot(1, 2, 1)
    splt.set_title(title, fontsize=14)
    splt.set_xlabel("Processing time (s)", fontsize=14)
    splt.set_ylabel("Test accuracy %", fontsize=14)
    splt.grid(True)
    splt.scatter(X, Y, color=color, marker='o', alpha=alpha, s=s, label=label)

    if print_front:
        O = np.array([(1-Y[i], X[i], 0) for i in range(len(X))])
        D = non_dominated_selection(O)
        X_d, Y_d = [X[d] for d in D], [Y[d] for d in D]
        sorted_args = np.argsort(Y_d)
        X_d = zip([X_d[i] for i in sorted_args], [X_d[i] for i in sorted_args[1:]] + [max(X_d)])
        X_d = [j for i in X_d for j in i]
        Y_d = zip([Y_d[i] for i in sorted_args], [Y_d[i] for i in sorted_args[:-1]] + [max(Y_d)])
        Y_d = [j for i in Y_d for j in i]
        splt.plot(X_d, Y_d, color=color, linestyle='--')
        splt.scatter(X_d, Y_d, color=color, marker=marker, s=40, alpha=1, label= label+" optimal")
    splt.legend()


def acc_params_subplot(Y, X, color, title="Solution space", label=None, print_front=False, alpha=0.2, s=5, marker="v"):
    splt = plt.subplot(1, 2, 2)
    splt.set_title(title, fontsize=14)
    splt.set_ylabel("Test accuracy %", fontsize=14)
    splt.set_xlabel("# Parameters", fontsize=14)
    splt.set_xscale("log")
    splt.grid(True)
    splt.scatter(X, Y, color=color, alpha=alpha, s=s, label=label)

    if print_front:
        O = np.array([(1-Y[i], X[i], 0) for i in range(len(X))])
        D = non_dominated_selection(O)
        X_d, Y_d = [X[d] for d in D], [Y[d] for d in D]
        sorted_args = np.argsort(Y_d)
        X_d = zip([X_d[i] for i in sorted_args], [X_d[i] for i in sorted_args[1:]]+[max(X_d)])
        X_d = [j for i in X_d for j in i]
        Y_d = zip([Y_d[i] for i in sorted_args], [Y_d[i] for i in sorted_args[:-1]] + [max(Y_d)])
        Y_d = [j for i in Y_d for j in i]
        splt.plot(X_d, Y_d, color=color, linestyle='--')
        splt.scatter(X_d, Y_d, color=color, marker=marker, s=40, alpha=1, label=label + " optimal")
    splt.legend()


def get_acc_time_param_from_results(R, phase="val"):
    acc = [r.test['system'].accuracy if phase == "test" else r.val['system'].accuracy for r in R.values()]
    time = [r.test['system'].time if phase == "test" else r.val['system'].time for r in R.values()]
    params = [r.test['system'].params if phase =="test" else r.test['system'].params for r in R.values()]
    return np.array(acc), np.array(time), np.array(params)


if __name__ == "__main__":

    plt.figure(figsize=(12, 5))
    plt.rcParams.update({'font.size': 12})

    # Information about the experiment
    experiment = 'bagging_boosting_of_chains_GA'
    experiment_dir = os.path.join(os.environ['FCM'], 'Examples', 'compute', experiment)
    meta_data_file = os.path.join(experiment_dir, 'results', 'metadata.json')

    ids = [9313361527064236, 7598351433769390]

    cmap = cm.get_cmap('jet')
    colors = ["#0066ff", "green", "red"]

    label = ["NSAS", "EARN+NSAS", "EARN"]
    markers = ["*", "D"]*2
    phase = 'test'

    for j, id in enumerate(ids):
        chain_data_path = os.path.join(experiment_dir, results_manager.get_results_by_id(meta_data_file, str(id)),
                                       'results_ensembles.pkl')
        dataset = results_manager.get_fieldval_by_id(meta_data_file, str(id), 'dataset')[0][12:-18]
        chain = io.read_pickle(chain_data_path)

        acc, time, params = get_acc_time_param_from_results(chain, phase)
        acc_params_subplot(acc, params, colors[j], "", label=label[j], print_front=True, marker=markers[j])
        acc_time_subplot(acc, time, colors[j], "",  label=label[j], print_front=True, marker=markers[j])

        if j == len(ids)-1:
            ref_nn = dict([(k, c) for k, c in chain.items() if len(c.test.keys()) < 3])
            acc_ref, time_ref, params_ref = get_acc_time_param_from_results(ref_nn, phase)
            acc_params_subplot(acc_ref, params_ref, "black", "", "DNNs", print_front=True, alpha=1, s=20)
            acc_time_subplot(acc_ref, time_ref, "black", "", "DNNs", print_front=True, alpha=1, s=20)

    plt.suptitle("Caltech256")
    plt.show()
