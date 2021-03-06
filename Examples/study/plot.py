import numpy as np
import Source.system_builder_serializable as sb
import Source.make_util as make
import os
import matplotlib.pyplot as plt
import Source.io_util as io

color = ["green", "yellow", "cyan", "purple", "pink", "blue", "gray", "black"]
marker_style = ["d", "o", "s"]
line_style = ["-", "--", ":", "-."]


def discretize_sort(X, Y, bins=100):
    X_ = np.zeros(bins)
    Y_ = [[] for i in range(bins)]
    max_X = max(X)
    min_X = min(X)
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        bin = int((x-min_X)/(max_X-min_X) * (bins-1))
        X_[bin] = (bin/(bins-1)) * (max_X - min_X) + min_X
        Y_[bin].append(y)

    D = [(X_[i], sum(Y_[i])/len(Y_[i]), np.std(Y_[i])) for i in range(bins) if len(Y_[i]) > 0]
    Y_avg = [d[1] for d in D]
    Y_std = [d[2] for d in D]
    X_ = [d[0] for d in D]
    return X_, Y_avg, Y_std


def plot_accuracy_parameters(record, system_color="green", label=None):
    plt.title("CIFAR-10 test set evaluation")
    plt.xlabel("# Parameters")
    plt.ylabel("Accuracy")
    plt.xscale("log")

    system_keys = [key for key in record.keys() if 'system' in key]
    pair_keys = [key for key in record.keys() if ';' in key]
    modified_keys = [key for key in record.keys() if key not in pair_keys and key not in system_keys]
    ref_keys = [key for key in record.keys() if
                key not in modified_keys and key not in system_keys and key not in pair_keys]

    X = [record[key]['system'].params for key in system_keys]
    Y = [record[key]['system'].accuracy for key in system_keys]
    plt.scatter(X, Y, color=system_color, s=10, alpha=0.75, label=label)

    X = [record[key]['system'].params for key in modified_keys]
    Y = [record[key]['system'].accuracy for key in modified_keys]
    plt.scatter(X, Y, color='black', s=10, alpha=1)

    X = [record[key]['system'].params for key in ref_keys]
    Y = [record[key]['system'].accuracy for key in ref_keys]
    plt.scatter(X, Y, color='black', s=10, alpha=1)


def plot_accuracy_time(record, system_color='green', label=None):

    plt.title("CIFAR-10 test set evaluation")
    plt.xlabel("Seconds")
    plt.ylabel("Accuracy")

    system_keys = [key for key in record.keys() if 'system' in key]
    pair_keys = [key for key in record.keys() if ';' in key]
    modified_keys = [key for key in record.keys() if key not in pair_keys and key not in system_keys]
    ref_keys = [key for key in record.keys() if
                key not in modified_keys and key not in system_keys and key not in pair_keys]

    X = [record[key]['system'].time for key in system_keys]
    Y = [record[key]['system'].accuracy for key in system_keys]
    plt.scatter(X, Y, color=system_color, s=10, alpha=1, label=label)

    X = [record[key]['system'].time for key in modified_keys]
    Y = [record[key]['system'].accuracy for key in modified_keys]
    plt.scatter(X, Y, color='black', marker='p', s=80, alpha=1, label=label)

    X = [record[key]['system'].time for key in ref_keys]
    Y = [record[key]['system'].accuracy for key in ref_keys]
    plt.scatter(X, Y, color='black', marker='p', s=80, alpha=1, label=label)


def plot_accuracy_parameters_time(record, title, system_color='blue', label=None, phase="test"):
    plt.subplot(1, 2, 1)
    plot_accuracy_time_old(record, title, system_color=system_color, label=label, phase=phase)
    plt.subplot(1, 2, 2)
    #plot_accuracy_parameters_old(record, title, system_color=system_color, label=label)


def plot_accuracy_time_old(record, title, system_color='blue', label=None, phase="test"):
    plt.title(title)
    plt.xlabel("Seconds")
    plt.ylabel("Accuracy")

    system_keys = [key for key in record.keys() if len(record[key].test) > 2]
    ref_keys = [key for key in record.keys() if key not in system_keys]

    X = [record[key].test['system'].time if phase == 'test' else record[key].val['system'].time for key in system_keys]
    Y = [record[key].test['system'].accuracy if phase == 'test' else record[key].val['system'].accuracy for key in system_keys]
    plt.scatter(X, Y, color=system_color, s=2, alpha=1, label=label)

    X = [record[key].test['system'].time if phase == 'test' else record[key].val['system'].time for key in ref_keys]
    Y = [record[key].test['system'].accuracy if phase == 'test' else record[key].val['system'].accuracy for key in ref_keys]
    plt.scatter(X, Y, color='black', edgecolors='gray', s=40, alpha=1)
    plt.legend()


def plot_accuracy_parameters_old(record, title, system_color="blue", label=None):
    plt.title(title)
    plt.xlabel("# Parameters")
    plt.ylabel("Accuracy")
    plt.xscale("log")

    system_keys = [key for key in record.keys() if len(record[key].test) > 2]
    ref_keys = [key for key in record.keys() if key not in system_keys]

    X = [record[key].test['system'].params for key in system_keys]
    Y = [record[key].test['system'].accuracy for key in system_keys]
    plt.scatter(X, Y, color=system_color, s=1, alpha=1, label=label)

    X = [record[key].test['system'].params for key in ref_keys]
    Y = [record[key].test['system'].accuracy for key in ref_keys]
    plt.scatter(X, Y, color='black', s=20)


def show():
    plt.legend()
    plt.show()


def save(f):
    plt.save_fig(f)


if __name__ == "__main__":
    method = 0
    plt.figure(0)
    toPlot_chain = io.read_pickle("./probability_threshold/results/imagenet/R_all")

    plt.figure(0)
    plot_accuracy_parameters(toPlot_chain, system_color='blue')
    #plot_accuracy_parameters(toPlot, system_color='green')

    plt.figure(1)
    plot_accuracy_time(toPlot_chain, system_color='blue')
    #plot_accuracy_time(toPlot, system_color='green')

    plt.show()
    # plot_accuracy_time(result_AVERAGE_4_models)
