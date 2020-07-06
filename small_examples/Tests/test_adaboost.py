import numpy as np
import Source.io_util as io
import os
import math


def adaboost_samme_label(L, gt):

    size = len(gt)
    w_instances = np.ones(int(size)) / size
    print(w_instances)
    alphas = []
    for im in range(L.shape[1]):
        # train classifier

        predictions = np.argmax(L[:, im, :], axis=1)
        K = len(np.unique(gt))
        fails = predictions != gt

        # Compute classifier error and its alpha term
        err = sum(w_instances[fails])/sum(w_instances)
        alpha = math.log((1-err)/err) + math.log(K-1)
        alphas.append(alpha)

        # Update w_instances
        for i in range(size):
            w_new = w_instances[i] * math.exp(alpha * (predictions[i] == gt[i]))
            w_instances[i] = w_new

        # Normalize
        w_instances = w_instances/sum(w_instances)

    return alphas


def adaboost_samme_logit(L, gt):

    size = len(gt)
    w_instances = np.ones(int(size)) / size
    alphas = []
    for im in range(L.shape[1]):
        predictions = np.argmax(L[:, im, :], axis=1)
        dividend = np.sum(np.exp(L[:, im, :]), axis=1)
        P = np.exp(L[:, im, :]) / dividend[:, None]
        max_probability = np.max(P, axis=1)
        diff = 1-max_probability

        K = len(np.unique(gt))
        fails = predictions != gt
        good = predictions == gt

        # Compute classifier error and its alpha term
        err = (sum(w_instances[fails])+sum(w_instances[good]*diff[good]))/sum(w_instances)
        alpha = math.log((1-err)/err) + math.log(K-1)
        alphas.append(alpha)

        # Update w_instances
        for i in range(size):
            w_new = w_instances[i] * math.exp(alpha) * (predictions[i] == gt[i])
            w_instances[i] = w_new

        # Normalize
        w_instances = w_instances/sum(w_instances)

    return alphas


def weighted_inference_logit(L, w_models):
    w_avg = np.average(L, axis=1, weights=w_models)
    return np.argmax(w_avg, axis=1)


def weighted_inference_label(L, w_models):
    P = np.argmax(L, axis=2)
    votes = np.apply_along_axis(np.bincount, 1, P, w_models, L.shape[2])
    predictions = np.argmax(votes, axis=1)
    return predictions


def weighted_max(Logits, w_models):
    Logits_mult = Logits * np.transpose([w_models])
    Logits_ = Logits.reshape(Logits_mult.shape[0], -1)
    predictions = np.apply_along_axis(lambda a: np.argmax(a) % Logits.shape[2], 1, Logits_)
    return predictions


if __name__ == "__main__":

    path = os.environ['FCM']
    Classifier_Path = "../../Definitions/Classifiers/sota_models_caltech256-32-dev_validation/"
    models = [Classifier_Path + f for f in os.listdir(Classifier_Path) if ".pkl" in f]

    # Train info
    r = np.random.randint(0, len(models))
    n_models = 3
    merge_models = models[r:r+n_models]
    size = 5e3
    gt_train = None
    L_all_train = None

    for i, m in enumerate(merge_models):
        c_dict = io.read_pickle(m)
        L, gt_train = c_dict['train']['logits'], c_dict['train']['gt']
        if L_all_train is None:
            L_all_train = np.empty((len(gt_train), len(merge_models), len(L[0])))
        L_all_train[:, i, :] = L

    # Test info
    size = 5e3
    gt = None
    L_all = None

    for i, m in enumerate(merge_models):
        c_dict = io.read_pickle(m)
        L, gt = c_dict['test']['logits'], c_dict['test']['gt']
        if L_all is None:
            L_all = np.empty((len(gt), len(merge_models), len(L[0])))
        L_all[:, i, :] = L

    w_models = adaboost_samme_logit(L_all_train, gt_train)
    predictions = weighted_inference_logit(L_all, w_models)
    predictions_avg = weighted_inference_logit(L_all, np.ones(n_models)/n_models)

    for i, m in enumerate(merge_models):
        pred = np.argmax(L_all[:, i, :], axis=1)
        acc = sum(pred == gt)/len(gt)
        print(m, acc, w_models[i])

    print("Accuracy:", sum(predictions == gt) / size)
    print("Accuracy average:", sum(predictions_avg == gt) / size)



