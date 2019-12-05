import Examples.paretto_front as front
import Source.io_util as io
import numpy as np
import matplotlib.pyplot as plt
import Examples.plot as myplt


def plot_accuracy_time(r,color="green"):
    cmap = plt.cm.get_cmap("Spectral")
    colors = cmap(np.linspace(0, 1, 10))

    plt.xlabel("Time (s)")
    plt.ylabel("Accuracy")
    plt.title("CIFAR-10")

    c = 0
    for key, val in r.items():
        for i in range(len(r[key][0])):
            acc = [x[i]['system'].accuracy for x in val]
            time = [x[i]['system'].time for x in val]
            plt.scatter(time, acc, color=colors[int(c/len(r.keys())*10)],  s=3)
        c = c+1

def plot_accuracy_instances(r):
    cmap = plt.cm.get_cmap("Spectral")
    colors = cmap(np.linspace(0, 1, 10))

    plt.xlabel("Instances bit model")
    plt.ylabel("Accuracy big model")
    plt.title("CIFAR10 trigger quality")

    for c, key in enumerate(r.keys()):
        for i in range(len(r[key][0])):
            tigger_results = r[key]

            max_acc = tigger_results[-1][i]['system'].accuracy
            min_acc = tigger_results[0][i]['system'].accuracy
            max_instances = tigger_results[-1][i]['big'].instances
            min_instances = tigger_results[0][i]['big'].instances

            acc_local = np.array([x[i]['system'].accuracy for x in tigger_results])
            acc_local = (acc_local - min_acc) / (max_acc - min_acc)

            instances_local = np.array([x[i]['big'].instances for x in tigger_results])
            instances_local = (instances_local - min_instances) / \
                              (max_instances - min_instances)

            plt.plot(instances_local, acc_local, color=colors[int(c/len(r.keys())*10)])


def plot_threshold_complexity(R):
    th_opt = []
    complexity = []
    plt.xscale("log")
    for key, val in R.items():
        acc = [r[0]['probability_threshold_trigger'].accuracy for r in val['2max_class']]
        th_opt = th_opt + [1/len(acc) * np.argmax(acc)]
        complexity = complexity + [val['2max_class'][0][0]['small'].params]
    plt.scatter(complexity, th_opt, s=10)


def sorted_components_params(R):
    tuple_sorted = sorted(R.items(), key=lambda item: item[1].params)
    return tuple_sorted


def results_models():
    import os
    import Source.system_builder as sb
    import Source.system_evaluator as eval
    import Source.make_util as make
    import Source.io_util as io

    R = {}

    Classifier_Path = "../../Definitions/Classifiers/"
    models = [f for f in os.listdir(Classifier_Path) if ".pkl" in f]

    for model in models:
        sys = sb.SystemBuilder(verbose=False)
        smallClassifier = make.make_classifier("c", Classifier_Path+model)
        sys.add_classifier(smallClassifier)
        R[Classifier_Path+model] = eval.evaluate(sys, "c").test['system']

    return R



if __name__ == "__main__":
    method = 0
    plot = False

    R = io.read_pickle("./results/cifar10/R.pkl")
    myplt.plot_accuracy_time(R)





