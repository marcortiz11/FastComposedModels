import Source.io_util as io
import Source.system_evaluator as eval
import Source.FastComposedModels_pb2 as fcm
import Source.system_evaluator_utils as eval_utils
import numpy as np
import math


def adaboost_samme_label(Logits, gt):
    size = len(gt)
    w_instances = np.ones(int(size)) / size
    alphas = []
    for im in range(Logits.shape[1]):

        # train classifier
        predictions = np.argmax(Logits[:, im, :], axis=1)
        K = len(np.unique(gt))
        fails = predictions != gt

        # Compute classifier error and its alpha term
        err = sum(w_instances[fails]) / sum(w_instances)
        err = max(1e-10, err)
        alpha = math.log((1 - err) / err) + math.log(K - 1)
        alphas.append(alpha)

        # Update w_instances
        for i in range(size):
            w_new = w_instances[i] * math.exp(alpha * (predictions[i] == gt[i]))
            w_instances[i] = w_new

        # Normalize
        w_instances = w_instances / sum(w_instances)

    return alphas


def adaboost_samme_logit(Logits, gt):
    size = len(gt)
    w_instances = np.ones(int(size)) / size
    alphas = []
    for im in range(Logits.shape[1]):
        predictions = np.argmax(Logits[:, im, :], axis=1)
        dividend = np.sum(np.exp(Logits[:, im, :]), axis=1)
        P = np.exp(Logits[:, im, :]) / dividend[:, None]
        max_probability = np.max(P, axis=1)
        diff = 1 - max_probability

        K = len(np.unique(gt))
        fails = predictions != gt
        good = predictions == gt

        # Compute classifier error and its alpha term
        err = (sum(w_instances[fails]) + sum(w_instances[good] * diff[good])) / sum(w_instances)
        err = max(1e-10, err)
        alpha = math.log((1 - err) / err) + math.log(K - 1)
        alphas.append(alpha)

        # Update w_instances
        for i in range(size):
            w_new = w_instances[i] * math.exp(alpha * (predictions[i] == gt[i]))
            w_instances[i] = w_new

        # Normalize
        w_instances = w_instances / sum(w_instances)

    return alphas


def weighted_average(Logits, w_models):
    w_avg = np.average(Logits, axis=1, weights=w_models)
    return np.argmax(w_avg, axis=1)


def weighted_voting(Logits, w_models):
    P = np.argmax(Logits, axis=2)
    votes = np.apply_along_axis(np.bincount, 1, P, w_models, Logits.shape[2])
    predictions = np.argmax(votes, axis=1)
    return predictions


def weighted_max(Logits, w_models):
    Logits_mult = Logits * np.transpose([w_models])
    Logits_ = Logits.reshape(Logits_mult.shape[0], -1)
    predictions = np.apply_along_axis(lambda a: np.argmax(a) % Logits.shape[2], 1, Logits_)
    return predictions


def evaluate(sys, results, c, check_classifiers, classifier_dict, input_ids=None, phase="test"):
    contribution = {}

    Logits = None
    GT = None
    c0 = sys.get(c.component_ids[0])
    c0_dict = io.read_pickle(c0.classifier_file)
    input_ids_ = input_ids if input_ids is not None else c0_dict[phase]['id']
    n_inputs = len(input_ids_)

    # For each classifier
    for i, c_id in enumerate(c.component_ids):
        component = sys.get(c_id)

        if check_classifiers:
            eval.check_valid_classifier(component)

        c_dict = io.read_pickle(component.classifier_file)
        L, gt, ids = eval_utils.get_Lgtid(c_dict, phase, input_ids)

        if GT is None: GT = gt
        else: assert np.array_equal(gt, GT),\
            "ERROR in Merger: Classifiers are supposed to have the same ids in the same order"

        if Logits is None:
            Logits = np.empty((len(gt), len(c.component_ids), len(L[0])))
        Logits[:, i, :] = L

        results[c_id] = eval.create_metrics_classifier(c_dict, np.argmax(L, axis=1), gt, n_inputs)
        eval.update_metrics_system(results, c_dict, n_inputs)

    # Apply the merge technique
    if c.merge_type == fcm.Merger.AVERAGE:
        avg = np.average(Logits, axis=1)
        pred = np.argmax(avg, axis=1)
    elif c.merge_type == fcm.Merger.VOTING:  # MAX VOTING
        P = np.argmax(Logits, axis=2)
        votes = np.apply_along_axis(np.bincount, 1, P, None, Logits.shape[2])
        pred = np.argmax(votes, axis=1)
    elif c.merge_type == fcm.Merger.MAX:  # GLOBAL MAX PROB CLASS
        Logits_ = Logits.reshape(Logits.shape[0], -1)
        pred = np.apply_along_axis(lambda a: np.argmax(a) % Logits.shape[2], 1, Logits_)
    elif c.merge_type > 2:
        # ADABOOSTING
        gt_train = None
        Logits_train = None
        for i, m in enumerate(c.component_ids):
            c_dict = io.read_pickle(m)
            L_train, gt_train = c_dict['train']['logits'], c_dict['train']['gt']
            if Logits_train is None:
                Logits_train = np.empty((len(gt_train), len(c.component_ids), len(L_train[0])))
            Logits_train[:, i, :] = L_train

        if c.merge_type == fcm.Merger.ADABOOST_LABEL_WEIGHTS_LOGIT_INFERENCE:
            w_models = adaboost_samme_label(Logits_train, gt_train)
            pred = weighted_average(Logits, w_models)
        elif c.merge_type == fcm.Merger.ADABOOST_LABEL_WEIGHTS_LABEL_INFERENCE:
            w_models = adaboost_samme_label(Logits_train, gt_train)
            pred = weighted_voting(Logits, w_models)
        elif c.merge_type == fcm.Merger.ADABOOST_LOGIT_WEIGHTS_LOGIT_INFERENCE:
            w_models = adaboost_samme_logit(Logits_train, gt_train)
            pred = weighted_average(Logits, w_models)
        elif c.merge_type == fcm.Merger.ADABOOST_LOGIT_WEIGHTS_LABEL_INFERENCE:
            w_models = adaboost_samme_logit(Logits_train, gt_train)
            pred = weighted_voting(Logits, w_models)
        elif c.merge_type == fcm.Merger.ADABOOST_LOGIT_WEIGHTS_MAX_INFERENCE:
            w_models = adaboost_samme_logit(Logits_train, gt_train)
            pred = weighted_max(Logits, w_models)
        elif c.merge_type == fcm.Merger.ADABOOST_LABEL_WEIGHTS_MAX_INFERENCE:
            w_models = adaboost_samme_label(Logits_train, gt_train)
            pred = weighted_max(Logits, w_models)

    assert classifier_dict is None, "Merger does not support being saved as a classifier yet!!!"
    contribution['model'] = dict(zip(input_ids_, list(c.component_ids)*n_inputs))
    contribution['predictions'] = dict(zip(ids, pred))
    contribution['gt'] = dict(zip(ids, gt))
    return contribution
