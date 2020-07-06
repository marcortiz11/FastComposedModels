import Source.system_evaluator as eval
import Source.FastComposedModels_pb2 as fcm
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


def get_logits_merged_components(sys, results, c, input_ids=None, check_classifiers=False, phase="test"):

    """
    Aranges in a 3-D Tensor the classification predictions of the merged ensembles/classifiers)
    :param sys: System
    :param results: Results Dictionary
    :param c: Merger component
    :param input_ids: Ids of the samples
    :param check_classifiers: Check classifiers during evaluation
    :param phase: Evaluation split (train,test,val)
    :return: The predictions, ground truth, and  evaluated ids of each merged component (ensemble/classifier)
    """

    # For each classifier
    Logits = None
    Gt = None
    Ids = None

    for i, c_id in enumerate(c.merged_ids):
        component = sys.get(c_id)

        if check_classifiers:
            eval.check_valid_classifier(component)

        contribution_component = eval.__evaluate(sys, results, component, check_classifiers, None,
                                            input_ids=input_ids, phase=phase)

        ids = np.array([key for key in contribution_component['gt'].keys()])
        gt = np.array([contribution_component['gt'][id] for id in ids])
        L = np.array([contribution_component['logits'][id] for id in ids])

        """
        c_dict = io.read_pickle(component.classifier_file)
        L, gt, ids = eval_utils.get_Lgtid(c_dict, phase, input_ids)
        """

        if Gt is None: Gt = gt
        else: assert np.array_equal(gt, Gt),\
            "ERROR in Merger: Classifiers are supposed to have the same ids in the same order"

        if Ids is None: Ids = ids
        else: assert np.array_equal(Ids, ids), \
            "ERROR in Merger: Merged solutions do not predict the same set of inputs"

        if Logits is None:
            Logits = np.empty((len(ids), len(c.merged_ids), len(L[0])))
        Logits[:, i, :] = L

        """
        results[c_id] = eval.create_metrics_classifier(c_dict, np.argmax(L, axis=1), gt, n_inputs)
        eval.update_metrics_system(results, c_dict, n_inputs)
        """

    return Logits, Gt, Ids


def evaluate(sys, results, c, check_classifiers, classifier_dict, input_ids=None, phase="test"):
    contribution = {}

    Logits, Gt , Ids = get_logits_merged_components(sys, results, c, check_classifiers=check_classifiers, phase=phase)

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
        # BOOSTING (ADA BOOST)

        train = dict()  # Fake variable just to obtain predictions during training
        train['system'] = eval.Metrics()
        Logits_train, Gt_train, _ = get_logits_merged_components(sys, train, c, check_classifiers=check_classifiers, phase="train")

        if c.merge_type == fcm.Merger.ADABOOST_LABEL_WEIGHTS_LOGIT_INFERENCE:
            w_models = adaboost_samme_label(Logits_train, Gt_train)
            pred = weighted_average(Logits, w_models)
        elif c.merge_type == fcm.Merger.ADABOOST_LABEL_WEIGHTS_LABEL_INFERENCE:
            w_models = adaboost_samme_label(Logits_train, Gt_train)
            pred = weighted_voting(Logits, w_models)
        elif c.merge_type == fcm.Merger.ADABOOST_LOGIT_WEIGHTS_LOGIT_INFERENCE:
            w_models = adaboost_samme_logit(Logits_train, Gt_train)
            pred = weighted_average(Logits, w_models)
        elif c.merge_type == fcm.Merger.ADABOOST_LOGIT_WEIGHTS_LABEL_INFERENCE:
            w_models = adaboost_samme_logit(Logits_train, Gt_train)
            pred = weighted_voting(Logits, w_models)
        elif c.merge_type == fcm.Merger.ADABOOST_LOGIT_WEIGHTS_MAX_INFERENCE:
            w_models = adaboost_samme_logit(Logits_train, Gt_train)
            pred = weighted_max(Logits, w_models)
        elif c.merge_type == fcm.Merger.ADABOOST_LABEL_WEIGHTS_MAX_INFERENCE:
            w_models = adaboost_samme_label(Logits_train, Gt_train)
            pred = weighted_max(Logits, w_models)

    assert classifier_dict is None, "Merger does not support being saved as a classifier yet!!!"
    contribution['model'] = dict(zip(Ids, list(c.merged_ids)*len(Ids)))
    contribution['predictions'] = dict(zip(Ids, pred))
    contribution['gt'] = dict(zip(Ids, Gt))

    return contribution
