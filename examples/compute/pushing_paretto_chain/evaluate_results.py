import examples.plot as myplt
import examples.paretto_front as front
import source.io_util as io
import numpy as np
import matplotlib.pyplot as plt
import pandas
import os

Classifier_Path = "../../definitions/Classifiers/"


def sorted_components_params(R):
    tuple_sorted = sorted(R.items(), key=lambda item: item[1].params)
    return tuple_sorted


def get_prediction_model_instance(front, id, input_ids):

    if len(input_ids) > 0:
        r = front[id]['system'] if "system" in id else front[Classifier_Path+id+'.pkl']['system']
        model_instance = dict([(i, r.instance_model[i]) for i in input_ids])

        simple_model = True
        for i, m in model_instance.items():
            simple_model = simple_model and len(m) == 1 and m[0] == id

        if simple_model:
            return model_instance
        else:
            input_ids_big = [i for i in input_ids if len(r.instance_model[i]) == 2]
            id_big = r.instance_model[input_ids_big[0]][1] if len(input_ids_big) > 0 else ""
            print("Big", id_big)
            pred_mod_inst_big = get_prediction_model_instance(front, id_big, input_ids_big)

            input_ids_small = input_ids
            id_small = r.instance_model[input_ids_small[0]][0]
            print("Small", id_small)
            pred_mod_inst_small = get_prediction_model_instance(front, id_small, input_ids_small)

            pred_mod_inst = dict([(k, v + pred_mod_inst_big[k] if k in pred_mod_inst_big else v)
                                    for k, v in pred_mod_inst_small.items()])
            return pred_mod_inst


def get_fronts(n, dataset):
    F = io.read_pickle("./results/"+dataset+"/models")
    for i in range(n+1):
        F.update(io.read_pickle("./results/"+dataset+"/front"+str(i)))
    return F


def plot_paretto_fronts(n, res_dir):
    toPlot = io.read_pickle(os.path.join(res_dir, "models"))
    plt.figure(0)
    plt.xscale("log")
    myplt.plot_accuracy_time(toPlot)

    toPlot = io.read_pickle(os.path.join(res_dir, "front0"))

    for i in range(1, n+1):
        toPlot = dict(("system_"+k, v) for k, v in io.read_pickle(os.path.join(res_dir, "front"+str(i))).items() if k not in toPlot)
        myplt.plot_accuracy_time(toPlot, system_color=myplt.color[i])
    plt.legend("Iteration 1", "Iteration 2")
    plt.show()


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


def print_global_info(F):
    max_acc_models = 0
    time_models = 0
    for k,v in F.items():
        if "system" not in k:
            if v['system'].accuracy > max_acc_models:
                max_acc_models = v['system'].accuracy
                time_models = v['system'].time

    max_acc_systems = 0
    time_systems = 0

    same_acc_systems = 1
    same_time_systems = 0

    for k, v in F.items():
        if "system" in k:
            if v['system'].accuracy > max_acc_systems:
                max_acc_systems = v['system'].accuracy
                time_systems = v['system'].time

            if v['system'].accuracy > max_acc_models \
                    and same_acc_systems > v['system'].accuracy:
                same_acc_systems = v['system'].accuracy
                same_time_systems = v['system'].time

    print("Max accuracy model:")
    print("\t Accuracy:" + str(max_acc_models))
    print("\t Time:" + str(time_models))
    print("Max accuracy system:")
    print("\t Accuracy:" + str(max_acc_systems))
    print("\t Time:" + str(time_systems))
    print("\t Speedup:" + str(time_models/time_systems))
    print("\t Accuracy increase:" + str(max_acc_systems - max_acc_models))
    print("Similar accuracy system:")
    print("\t Accuracy:" + str(same_acc_systems))
    print("\t Time:" + str(same_time_systems))
    print("\t Speedup:" + str(time_models / same_time_systems))
    print("\t Accuracy increase:" + str(same_acc_systems - max_acc_models))


if __name__ == "__main__":
    method = 0

    F = get_fronts(2, 'cifar10')

    # study 0: Print paretto fronts
    res_dir = './results/cifar10/'
    plot_paretto_fronts(2, res_dir)

    # study 1: Plot execution instances per model
    # plot_instances_per_model(F, 4, n_models_plot=3)

    # study 2: Global statistics
    print_global_info(F)

    # study 2: Plot prediction instances per model
    # plot_predictions_per_model(F)



