import Examples.plot as myplt
import Examples.paretto_front as front
import Source.io_util as io
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_increase_accuracy_old(results_location):

    color = ["green", "yellow", "cyan", "purple", "pink", "blue", "gray", "black"]
    line_style = ["-", "--"]
    technique = ["BAGGING_AVERAGE",
                 "BAGGING_MAX",
                 "BAGGING_VOTING",
                 "BOOST_LABEL_WEIGHTS_AVERAGE",
                 "BOOST_LABEL_WEIGHTS_VOTING",
                 "BOOST_LOGIT_WEIGHTS_VOTING",
                 "BOOST_LOGIT_WEIGHTS_AVERAGE",
                 "ADABOOST_LOGIT_WEIGHTS_MAX_INFERENCE",
                 "ADABOOST_LABEL_WEIGHTS_MAX_INFERENCE",]

    plt.figure(0)
    plt.title("Increment in accuracy")
    plt.xlabel("Acc bigger - Acc smaller")
    plt.ylabel("Average increment accuracy")
    plt.figure(1)
    plt.title("STD")
    plt.xlabel("Acc bigger - Acc smaller")
    plt.ylabel("STD increment in accuracy")
    plt.figure(2)
    plt.title("Percentage")
    plt.xlabel("Acc bigger - Acc smaller")
    plt.ylabel("Percentage that increase accuracy")

    for im, models in enumerate([3, 5]):
        for merge_technique in [0, 1, 2]:
            results = {}
            print("merge:"+str(merge_technique), "model:", str(models))
            for part in range(2):
                fname = results_location+"/R_"+str(merge_technique)+"_"+str(models)+"_part"+str(part)
                if os.path.exists(fname):
                    results.update(io.read_pickle(fname))

            precision = int(500)
            X = [[] for i in range(precision)]
            # X_count = np.ones(precision)

            for key, val in results.items():

                max_acc = 0
                min_acc = 1
                avg_acc = 0

                if "system" in key or ";" in key:

                    for key2, val2 in val.items():
                        if val2.accuracy > max_acc and "system" not in key2:
                            max_acc = val2.accuracy
                        elif val2.accuracy < min_acc:
                            min_acc = val2.accuracy
                        elif "system" not in key2:
                            avg_acc += val2.accuracy
                    avg_acc /= len(val.items())-1
                    increase = val['system'].accuracy / max_acc

                    bin = int((max_acc-min_acc)*precision)
                    X[bin].append(increase)

            std = [np.std(X[i]) for i in range(precision) if len(X[i]) > 0]
            better = [sum(np.array(X[i]) > 1)/len(X[i]) for i in range(precision) if len(X[i]) > 0]
            D = [(i, sum(X[i])/len(X[i])) for i in range(precision) if len(X[i]) > 0]
            X_avg = [d[1] for d in D]
            Y = [d[0]/precision for d in D]

            plt.figure(0)
            plt.plot(Y, X_avg, color=color[merge_technique], linestyle=line_style[im], label=technique[merge_technique]+"_"+str(models)+"models")
            plt.legend()

            plt.figure(1)
            plt.plot(Y, std, color=color[merge_technique], linestyle=line_style[im], label=technique[merge_technique]+"_"+str(models)+"models")
            plt.legend()

            plt.figure(2)
            plt.plot(Y, better, color=color[merge_technique], linestyle=line_style[im],
                     label=technique[merge_technique] + "_" + str(models) + "models")
            plt.legend()

    plt.show()


def plot_increase_accuracy(results_location):

    color = ["green", "yellow", "cyan", "purple", "pink", "blue", "gray", "black"]
    line_style = ["-", "--"]
    technique = ["BAGGING_AVERAGE",
                 "BAGGING_MAX",
                 "BAGGING_VOTING",
                 "BOOST_LABEL_WEIGHTS_AVERAGE",
                 "BOOST_LABEL_WEIGHTS_VOTING",
                 "BOOST_LOGIT_WEIGHTS_VOTING",
                 "BOOST_LOGIT_WEIGHTS_AVERAGE",
                 "ADABOOST_LOGIT_WEIGHTS_MAX_INFERENCE",
                 "ADABOOST_LABEL_WEIGHTS_MAX_INFERENCE", ]

    plt.figure(0)
    plt.title("Increment in accuracy")
    plt.xlabel("Acc bigger - Acc smaller")
    plt.ylabel("Average increment accuracy")
    plt.figure(1)
    plt.title("STD")
    plt.xlabel("Acc bigger - Acc smaller")
    plt.ylabel("STD increment in accuracy")
    plt.figure(2)
    plt.title("Percentage")
    plt.xlabel("Acc bigger - Acc smaller")
    plt.ylabel("Percentage that increase accuracy")

    for im, models in enumerate([3]):
        for merge_technique in [0, 1, 2, 3, 4, 5, 6]:
            results = {}
            print("merge:"+str(merge_technique), "model:", str(models))
            for part in range(2):
                fname = results_location+"/R_"+str(merge_technique)+"_"+str(models)+"_part"+str(part)+".pkl"
                if os.path.exists(fname):
                    results.update(io.read_pickle(fname))

            X = []
            Y = []

            for key, val in results.items():
                max_acc = 0
                min_acc = 1

                if "system" in key:  # Composed model
                    for key2, val2 in val.items():
                        if "system" not in key2:  # Individual components
                            if val2.accuracy > max_acc:
                                max_acc = val2.accuracy
                            if val2.accuracy < min_acc:
                                min_acc = val2.accuracy

                    increase = val['system'].accuracy / max_acc
                    Y.append(increase)
                    X.append(max_acc-min_acc)

            X_, Y_avg, Y_std = myplt.discretize_sort(X, Y, bins=50)

            plt.figure(0)
            plt.plot(X_, Y_avg, color=color[merge_technique], linestyle=line_style[im], label=technique[merge_technique]+"_"+str(models)+"models")
            plt.legend()

            plt.figure(1)
            plt.plot(X_, Y_std, color=color[merge_technique], linestyle=line_style[im], label=technique[merge_technique]+"_"+str(models)+"models")
            plt.legend()

            """
            plt.figure(2)
            plt.plot(Y, better, color=color[merge_technique], linestyle=line_style[im],
                     label=technique[merge_technique] + "_" + str(models) + "models")
            plt.legend()
            """

    plt.show()

def plot_params_accuracy_front(out_dir):
    import os

    plt.figure(3)
    plt.title("Parameters of the solutions")
    plt.xlabel("# Params")
    plt.xscale("log")
    plt.ylabel("Accuracy")

    for im, models in enumerate([3, 5]):
        for merge_technique in [0, 1, 2]:
            results = {}
            print("merge:"+str(merge_technique), "model:", str(models))
            for part in range(2):
                fname = os.path.join(out_dir,
                                    "R_"+str(merge_technique)+"_"+str(models)+"_part"+str(part)+".pkl")
                if os.path.exists(fname):
                    results.update(io.read_pickle(fname))
            myplt.plot_accuracy_parameters_old(results, myplt.color[merge_technique])

    plt.legend()
    plt.show()



def sorted_components_params(R):
    tuple_sorted = sorted(R.items(), key=lambda item: item[1].params)
    return tuple_sorted


if __name__ == "__main__":
    method = 0
    plot = False

    res = "./results/flowers102-32-dev"
    # plot_params_accuracy_front(res)
    plot_increase_accuracy(res)

    if plot:
        toPlot = io.read_pickle("./results/cifar10/R_2_3_part0")
        toPlot.update(io.read_pickle("./results/cifar10/R_2_3_part1"))
        # toPlot.update(io.read_pickle("./results/cifar10/R_2_3_part2"))
        myplt.plot_accuracy_time_old(toPlot, system_color='blue')

        toPlot = io.read_pickle("./results/cifar10/R_1_3_part0")
        toPlot.update(io.read_pickle("./results/cifar10/R_1_3_part1"))
        # toPlot.update(io.read_pickle("./results/cifar10/R_1_3_part2"))
        myplt.plot_accuracy_time_old(toPlot, system_color='blue')

        toPlot = io.read_pickle("./results/cifar10/R_0_3_part0")
        toPlot.update(io.read_pickle("./results/cifar10/R_0_3_part1"))
        # toPlot.update(io.read_pickle("./results/cifar10/R_0_3_part1"))
        myplt.plot_accuracy_time_old(toPlot, system_color='blue')

        toPlot = io.read_pickle("./results/cifar10/R_2_5_part0")
        toPlot.update(io.read_pickle("./results/cifar10/R_2_5_part1"))
        toPlot.update(io.read_pickle("./results/cifar10/R_2_5_part2"))
        #toPlot.update(io.read_pickle("./results/R_2_5_part3"))
        myplt.plot_accuracy_time_old(toPlot, system_color='blue')

        toPlot = io.read_pickle("./results/cifar10/R_1_5_part0")
        toPlot.update(io.read_pickle("./results/cifar10/R_1_5_part1"))
        toPlot.update(io.read_pickle("./results/cifar10/R_1_5_part2"))
        toPlot.update(io.read_pickle("./results/cifar10/R_1_5_part3"))
        myplt.plot_accuracy_time_old(toPlot, system_color='blue')

        toPlot = io.read_pickle("./results/cifar10/R_0_5_part0")
        toPlot.update(io.read_pickle("./results/cifar10/R_0_5_part1"))
        toPlot.update(io.read_pickle("./results/cifar10/R_0_5_part2"))
        toPlot.update(io.read_pickle("./results/cifar10/R_0_5_part3"))
        myplt.plot_accuracy_time_old(toPlot, system_color='blue', label="Classic Ensemble")

        toPlot = io.read_pickle("../probability_threshold/results/cifar10/R")
        myplt.plot_accuracy_time(toPlot, system_color='green', label="Chain 2 models")

        myplt.show()

        """
        toPlot = io.read_pickle("./results/imagenet/R_0_3_part0")
        myplt.plot_accuracy_parameters(toPlot, system_color='green')
        toPlot = io.read_pickle("./results/imagenet/R_1_3_part0")
        myplt.plot_accuracy_parameters(toPlot, system_color='yellow')
        toPlot = io.read_pickle("./results/imagenet/R_2_3_part0")
        myplt.plot_accuracy_parameters(toPlot, system_color='cyan')
        myplt.show()
        """

    R = {}

    for m in range(3):
        for part in range(3):
            for n_models in [3]:
               R.update(io.read_pickle("./results/R_"+str(m)+"_"+str(n_models)+"_part"+str(part)))

    front_models = front.get_front_params_accuracy(R)
    worst_models = front.get_front_accuracy_params(R)
    print("FRONT SOLUTIONS:")
    print('\t\n'.join(front_models))
    print("\n WORST SOLUTIONS:")
    print('\t\n'.join(worst_models))

    # QUESTION 1: How many times is the complex model bigger than the simplest model?
    times_bigger = np.array([])
    for i,model in enumerate(front_models):
        if "system" in model:
            system_sorted = sorted_components_params(R[model])
            times_bigger = np.append(times_bigger, system_sorted[-1][1].params / system_sorted[0][1].params)

    print(times_bigger)

    # QUESTION 1: How many times is the complex model bigger than the simplest model?
    times_bigger = np.array([])
    for i, model in enumerate(worst_models):
        if "system" in model:
            system_sorted = sorted_components_params(R[model])
            times_bigger = np.append(times_bigger, system_sorted[-1][1].params / system_sorted[0][1].params)

    print(times_bigger)




