import Source.system_builder as sb
import Source.system_evaluator as eval
import Source.protobuf.make_util as make
import Source.io_util as io
import numpy as np


def build_train_trigger(model1_dict, th):
    classifier_trigger_dict = {}

    performance = make.make_performance_metrics(**{})

    #Train dict
    L = model1_dict['train']['logits']
    dividend = np.sum(np.exp(L), axis=1)
    P = np.exp(L) / dividend[:, None]
    sort = np.sort(P, axis=1)  # Sort class probabilities
    diff = sort[:, -1] - sort[:, -2]  # Difference
    logits_trigger = np.empty((diff.shape[0], 2))
    logits_trigger[:, 0] = diff < th
    logits_trigger[:, 1] = diff >= th

    pred_model1 = np.argmax(L, axis=1)
    gt_model1 = model1_dict['train']['gt']

    train = make.make_classifier_raw_data(logits_trigger,(pred_model1 == gt_model1), np.copy(model1_dict['train']['id']))

    # Test dict
    L = model1_dict['test']['logits']
    dividend = np.sum(np.exp(L), axis=1)
    P = np.exp(L) / dividend[:, None]
    sort = np.sort(P, axis=1)  # Sort class probabilities
    diff = sort[:, -1] - sort[:, -2]  # Difference
    logits_trigger = np.empty((diff.shape[0], 2))
    logits_trigger[:, 0] = diff < th
    logits_trigger[:, 1] = diff >= th

    pred_model1 = np.argmax(L, axis=1)
    gt_model1 = model1_dict['test']['gt']

    test = make.make_classifier_raw_data(logits_trigger, (pred_model1 == gt_model1), np.copy(model1_dict['test']['id']))

    classifier_trigger_dict = make.make_classifier_dict("trigger_classifier", "cifar10", train, test, performance)
    io.save_pickle('./trigger_tmp', classifier_trigger_dict)
    classifier = make.make_classifier("trigger_classifier", "./trigger_tmp")
    return classifier


if __name__ == "__main__":

    small_cfile = "../Definitions/Classifiers/V001_DenseNet_s1_39"
    big_cfile = "../Definitions/Classifiers/V001_DenseNet_s1_80"
    c_dict_big = io.read_pickle(big_cfile)
    c_dict_small = io.read_pickle(small_cfile)
    th = 1.1

    # Building system skeleton
    sys = sb.SystemBuilder(verbose=False)
    big_classifier = make.make_classifier("big", big_cfile)
    sys.add_classifier(big_classifier)
    c = build_train_trigger(c_dict_small, th)
    trigger = make.make_trigger("trigger", c, ["big"])
    sys.add_trigger(trigger)
    small_classifier = make.make_classifier("small", small_cfile, "trigger")
    sys.add_classifier(small_classifier)

    # Test that times are consistent
    classifier, evaluation = sys.build_classifier_dict("built", "small")
    import Source.system_evaluator_utils as eval_utils
    eval_utils.pretty_print(evaluation)

    # assert (classifier['test']['time_instance'] == (c_dict_small['metrics']['time'] + c_dict_big['metrics']['time']/128.0)).all()
    print("All good with time!")
    acc_test = sum(np.argmax(classifier['test']['logits'], axis=1) == classifier['test']['gt'])/1e4
    assert (acc_test == evaluation.test['system'].accuracy)
    print("All good with accuracy!")
    assert (classifier['metrics']['params'] == evaluation.test['system'].params)
    print("All good with params!")
