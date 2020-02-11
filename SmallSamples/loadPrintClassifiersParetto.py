import numpy as np
import Source.system_builder as sb
import Source.make_util as make
import Source.system_evaluator as eval
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import random


def plot_accuracy_parameters_front_line(front, color='black', label="Solution set 1"):
    plt.title("Solution Space")
    plt.xlabel("Resource")
    plt.ylabel("Performance")
    plt.xscale("log")
    X = []
    Y = []
    for i in front.items():
        x = i[1]['system'].params
        X.append(x)
        X.append(x)
        y = i[1]['system'].accuracy
        Y.append(y)
        Y.append(y)
    X = X[1:]
    Y = Y[:-1]
    plt.plot(X, Y, color=color, label=label)

def plot_accuracy_parameters(record, color='black', label="Solution set 1"):
    plt.title("Solution Space")
    plt.xlabel("Resource")
    plt.ylabel("Performance")
    plt.xscale("log")
    X = []
    Y = []
    for key in record.keys():
        X.append(record[key].test['system'].params)
        Y.append(record[key].test['system'].accuracy)
    plt.scatter(X, Y, color=color, s=20, label=label)

def plot_accuracy_time(record, color='black', label="Solution set 1"):
    plt.title("Solution Space")
    plt.xlabel("Resource")
    plt.ylabel("Performance")
    plt.grid(True, alpha=0.3)
    #plt.xscale("log")
    X = []
    Y = []
    for key in record.keys():
        X.append(record[key].test['system'].time)
        Y.append(record[key].test['system'].accuracy)
    plt.scatter(X, Y, color=color, s=20, label=label)


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
    dsets = ["front45_models",
             "sota_models_cifar100-32-dev_validation",
             "sota_models_cifar10-32-dev_validation",
             "sota_models_fashion-mnist-32-dev_validation",
             "sota_models_flowers102-32-dev_validation",
             "sota_models_gtsrb-32-dev_validation",
             "sota_models_gtsrbcrop-32-dev_validation",
             "sota_models_mnist-32-dev_validation",
             "sota_models_stl10-32-dev_validation",
             "sota_models_caltech256-32-dev_validation",
             "sota_models_svhn-32-dev_validation",
             "sota_models_quickdraw-28-dev_validation"]

    for id, d in enumerate(dsets):
        Classifier_Path = Dataset_Path + d + '/'
        models = [f for f in os.listdir(Classifier_Path)]
        print(models)

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

            records[m_] = results
            # print(results.test)

        import Source.io_util as io
        if not os.path.exists('./models_evaluation/%s'%d):
            os.makedirs('./models_evaluation/%s'%d)
        io.save_pickle('./models_evaluation/%s/models.pkl'%d, records)

        import Examples.paretto_front as pareto
        front = pareto.get_front_time_accuracy(records, phase="test")

        plot_accuracy_time(front)
        plt.legend()
        plt.show()

