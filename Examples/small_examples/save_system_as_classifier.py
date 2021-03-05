import Source.protobuf.make_util as make
import Source.system_builder as sb
import Source.system_evaluator as eval
import Source.io_util as io
import numpy as np


import os
Classifier_Path = "../../Definitions/Classifiers/"
models = [Classifier_Path+f for f in os.listdir(Classifier_Path) if ".pkl" in f][::10]


def build_train_trigger(model1_dict, th, name="trigger"):
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
    io.save_pickle('../Definitions/Triggers/Tmp/'+name, classifier_trigger_dict)
    classifier = make.make_classifier("trigger_classifier", "../Definitions/Triggers/Tmp/"+name)
    return classifier


net1 = 'system_DenseNet201_cifar10.pkl_ResNet152_cifar10.pklth=0.7'
net2 = 'GoogleNet_cifar10.pkl'
m = Classifier_Path + net1
m_ = Classifier_Path + net2
th = 0.7

sys = sb.SystemBuilder(verbose=True)

name2 = m_
bigClassifier = make.make_empty_classifier("big")
bigClassifier.classifier_file = name2
sys.add_classifier(bigClassifier)

trigger = make.make_trigger("probability_threshold_trigger", make.make_empty_classifier(), ["big"])
sys.add_trigger(trigger)

name1 = m
smallClassifier = make.make_empty_classifier("small", "probability_threshold_trigger")
model1_dict = io.read_pickle(name1, suffix="", verbose=False)
smallClassifier.classifier_file = name1
sys.add_classifier(smallClassifier)

classifier_trigger = build_train_trigger(model1_dict, th)
trigger = make.make_trigger("probability_threshold_trigger", classifier_trigger, ["big"])
trigger.id = "probability_threshold_trigger"
sys.replace(trigger.id, trigger)

metrics = eval.evaluate(sys, "small", check_classifiers=False)
eval.pretty_print(metrics)

# cl = sys.build_classifier_dict("small")
# io.save_pickle(Classifier_Path+"system_"+net1+"_"+net2+"th=0.7", cl)
