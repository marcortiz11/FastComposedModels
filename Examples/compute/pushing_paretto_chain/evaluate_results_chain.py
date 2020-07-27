import Examples.study.plot as myplt
import Source.io_util as io
import Examples.study.paretto_front as paretto
import numpy as np
import matplotlib.pyplot as plt


def plot_instances_per_model(F, n_models, n_models_plot=3):
    plt.figure(0)
    plt.xlabel("Model accuracy")
    plt.ylabel("# Inference")
    plt.title("Instances per model")
    plt.yscale("log")

    F_nm = dict([(k,v) for k,v in F.items() if "num" in F[k] and F[k]["num"] == n_models])
    print(F_nm.keys())
    rand_numb = np.random.randint(0, len(F_nm.keys()), n_models_plot)
    print(rand_numb)

    for i,n in enumerate(rand_numb):
        id = list(F_nm.keys())[n]
        input_ids = list(F_nm[id]['system'].instance_model.keys())
        d = get_prediction_model_instance(F, id, input_ids)
        hist = {}
        hist_pred = {}
        for k, v in d.items():
            if v[-1] not in hist_pred: hist_pred[v[-1]] = 1
            else: hist_pred[v[-1]] += 1
            for k_ in v:
                if k_ not in hist:
                    hist[k_] = 1
                else:
                    hist[k_] += 1

        labels = np.array([k for k, v in hist.items()])
        Y = np.array([hist[k] for k in labels])
        Y_ = np.array([hist_pred[k] for k in labels])
        X = np.array([F[Classifier_Path+k+'.pkl']['system'].accuracy for k in hist.keys()])
        X_arg = np.argsort(X)
        #Y = Y[X_arg]
        #X = X[X_arg]
        #Y_ = Y_[X_arg]
        labels = str(F[id]['system'].accuracy) + " " + str(F[id]['system'].time)
        plt.plot(X, Y, label=labels, color=myplt.color[i])
        plt.plot(X, Y_, linestyle="--", color=myplt.color[i])
        plt.legend()


def plot_paretto_fronts(n):
    toPlot = io.read_pickle("./results/models")
    myplt.plot_accuracy_time(toPlot)

    for i in range(n+1):
        print(i)
        toPlot = dict((k, v) for k, v in io.read_pickle("./results/front"+str(i)).items() if k not in toPlot)
        myplt.plot_accuracy_time(toPlot, system_color=myplt.color[i])

    """
    toPlot = dict((k, v) for k, v in io.read_pickle("./results/front2").items() if k not in toPlot)
    myplt.plot_accuracy_time(toPlot, system_color='green')

    toPlot = dict((k, v) for k, v in io.read_pickle("./results/front3").items() if k not in toPlot)
    myplt.plot_accuracy_time(toPlot, system_color='cyan')
    """


if __name__ == "__main__":

    front_chain = io.read_pickle("./results/example_chain/R_all")
    front_chain = paretto.get_front_time_accuracy(front_chain)
    plot_paretto_fronts(2)
    myplt.plot_accuracy_time(front_chain, system_color=myplt.color[3])
    myplt.show()
    plot_instances_per_model(front_chain, 2)
