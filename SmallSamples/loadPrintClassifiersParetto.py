import numpy as np
import Source.system_builder as sb
import Source.make_util as make
import Source.system_evaluator as eval
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time


def plot_accuracy_parameters(record):
    plt.title("Solution Space")
    plt.xlabel("Resource")
    plt.ylabel("Performance")
    plt.xscale("log")
    for key in record.keys():
        x = record[key]['system'].params
        y = record[key]['system'].accuracy
        if '_vs_' in key:  # Composed models
            th = float(key[key.find('=')+1:])
            cmap = cm.get_cmap("jet")
            colors = cmap(np.linspace(0, 1, 10))
            plt.scatter(x, y, color=colors[int(th*10)], s=10)
        else:
            color = 'red'
            plt.scatter(x, y, color=color, s=40)
        # plt.annotate(key, (x, y))
    plt.show()


def plot_accuracy_time(record):
    plt.title("Solution Space")
    plt.xlabel("Resource")
    plt.ylabel("Performance")
    plt.xscale("log")
    for key in record.keys():
        x = record[key].time
        y = record[key].accuracy
        if '_vs_' in key:
            th = float(key[key.find('=') + 1:])
            cmap = cm.get_cmap("jet")
            colors = cmap(np.linspace(0, 1, 10))
            plt.scatter(x, y, color=colors[int(th * 10)], s=10)
        else:
            color = 'black' if 'modified' in key else 'red'
            plt.scatter(x, y, color=color, s=40)
        # plt.annotate(key, (x, y))
    plt.show()


def plot_accuracy_dataset(record, dataset):
    plt.title("Model accuracy all datasets")
    plt.xlabel("Dataset")
    plt.ylabel("Accuracy")
    for key in record.keys():
        x = dataset
        y = record[key].accuracy
        if '_vs_' in key:
            th = float(key[key.find('=') + 1:])
            cmap = cm.get_cmap("jet")
            colors = cmap(np.linspace(0, 1, 10))
            plt.scatter(x, y, color=colors[int(th * 10)], s=10)
        else:
            color = 'black' if 'modified' in key else 'red'
            plt.scatter(x, y, color=color, s=40)
    plt.show()


if __name__ == "__main__":
    # Get the CIFAR-100 models form the Classifiers dir
    import os
    Dataset_Path = "../Definitions/Classifiers/"
    dsets = ["cifar100-32-dev",
             "cifar10-32-dev",
             "fashion-mnist-32-dev",
             "flowers102-32-dev",
             # "flowers-32-dev",        # error, amount of samples
             "food101-32-dev",
             "gtsrb-32-dev",
             "gtsrbcrop-32-dev",
             "mnist-32-dev",
             "stl10-32-dev",
             "caltech256-32-dev",
             # "textures_32.h5",        # error on submit
             # "indoor67_32.h5",
             # "places_32.h5",
             "svhn-32-dev",
             "quickdraw-28-dev"]

    dsets=["cifar10"]

    for id, d in enumerate(dsets):
        Classifier_Path = Dataset_Path + d + '/'
        models = [f for f in os.listdir(Classifier_Path) if ".pkl" in f]

        # Creating system
        sys = sb.SystemBuilder(verbose=False)
        smallClassifier = make.make_empty_classifier("Classifier")
        sys.add_classifier(smallClassifier)
        records = {}

        for m_ in models:

            # Model 2
            name2 = Classifier_Path + m_
            model2 = make.make_empty_classifier()
            model2.id = "Classifier"
            model2.classifier_file = name2
            sys.replace(model2.id, model2)

            evaluate_time_start = time.time()
            results = eval.evaluate(sys, model2.id)
            eval_time = time.time() - evaluate_time_start

            print("Evaluation time:", eval_time)

            records[m_] = results.test
            # print(results.test)

        import Examples.paretto_front as pareto
        front = pareto.get_front_params_accuracy(records)
        plot_accuracy_parameters(front)

