import Source.io_util as io
import Source.protobuf.make_util as make
import Source.system_builder as sb
import Source.system_evaluator as eval
import numpy as np


def build_train_trigger(model1_dict, th):
    classifier_trigger_dict = {}

    performance = make.make_performance_metrics(**{})

    # Train dict
    L = model1_dict['train']['logits']
    dividend = np.sum(np.exp(L), axis=1)
    P = np.exp(L) / dividend[:, None]
    sort = np.sort(P, axis=1)  # Sort class probabilities
    diff = sort[:, -1] - sort[:, -2]  # Difference
    logits_trigger = np.empty((diff.shape[0], 2))
    logits_trigger[:, 0] = diff < th
    logits_trigger[:, 1] = diff >= th

    pred_model1 = np.argmax(L,axis=1)
    gt_model1 = model1_dict['train']['gt']

    train = make.make_classifier_raw_data(logits_trigger,(pred_model1 == gt_model1), np.copy(model1_dict['train']['id']))

    # Test dict
    L = model1_dict['test']['logits']
    dividend = np.sum(np.exp(L), axis=1)
    P = np.exp(L)/dividend[:, None]
    sort = np.sort(P, axis=1)  # Sort class probabilities
    diff = sort[:, -1] - sort[:, -2]  # Difference
    logits_trigger = np.empty((diff.shape[0], 2))
    logits_trigger[:, 0] = diff < th
    logits_trigger[:, 1] = diff >= th

    pred_model1 = np.argmax(L, axis=1)
    gt_model1 = model1_dict['test']['gt']

    test = make.make_classifier_raw_data(logits_trigger, (pred_model1 == gt_model1), np.copy(model1_dict['test']['id']))

    classifier_trigger_dict = make.make_classifier_dict("trigger_classifier", "cifar10", train, test, performance)
    io.save_pickle('../../Definitions/Classifiers/tmp/trigger_random_threshold.pkl', classifier_trigger_dict)
    classifier = make.make_classifier("trigger_classifier", "../../Definitions/Classifiers/tmp/trigger_random_threshold.pkl")
    return classifier


def build_train_trigger2(model1_dict, th):
    classifier_trigger_dict = {}

    performance = make.make_performance_metrics(**{})

    # Train dict
    L = model1_dict['train']['logits']
    dividend = np.sum(np.exp(L), axis=1)
    P = np.exp(L) / dividend[:, None]
    max_P = np.max(P, axis=1)
    logits_trigger = np.empty((max_P.shape[0], 2))
    logits_trigger[:, 0] = max_P < th
    logits_trigger[:, 1] = max_P >= th

    pred_model1 = np.argmax(L, axis=1)
    gt_model1 = model1_dict['train']['gt']

    train = make.make_classifier_raw_data(logits_trigger,(pred_model1 == gt_model1), np.copy(model1_dict['train']['id']))

    # Test dict
    L = model1_dict['test']['logits']
    dividend = np.sum(np.exp(L), axis=1)
    P = np.exp(L)/dividend[:, None]
    max_P = np.max(P, axis=1)
    logits_trigger = np.empty((max_P.shape[0], 2))
    logits_trigger[:, 0] = max_P < th
    logits_trigger[:, 1] = max_P >= th

    pred_model1 = np.argmax(L, axis=1)
    gt_model1 = model1_dict['test']['gt']

    test = make.make_classifier_raw_data(logits_trigger, (pred_model1 == gt_model1), np.copy(model1_dict['test']['id']))

    classifier_trigger_dict = make.make_classifier_dict("trigger_classifier", "cifar10", train, test, performance)
    io.save_pickle('../../Definitions/Classifiers/tmp/trigger_random_threshold', classifier_trigger_dict)
    classifier = make.make_classifier("trigger_classifier", "../../Definitions/Classifiers/tmp/trigger_random_threshold")
    return classifier


def build_train_trigger3(model1_dict, p):
    classifier_trigger_dict = {}

    performance = make.make_performance_metrics(**{})

    # Train dict
    L = model1_dict['train']['logits']
    logits_trigger = np.empty((L.shape[0], 2))
    logits_trigger[:, 0] = np.random.binomial(1, p, L.shape[0])
    logits_trigger[:, 1] = 1 - logits_trigger[:, 0]

    pred_model1 = np.argmax(L, axis=1)
    gt_model1 = model1_dict['train']['gt']

    train = make.make_classifier_raw_data(logits_trigger,(pred_model1 == gt_model1), np.copy(model1_dict['train']['id']))

    # Test dict
    L = model1_dict['test']['logits']
    logits_trigger = np.empty((L.shape[0], 2))
    logits_trigger[:, 0] = np.random.binomial(1, p, L.shape[0])
    logits_trigger[:, 1] = 1 - logits_trigger[:, 0]

    pred_model1 = np.argmax(L, axis=1)
    gt_model1 = model1_dict['test']['gt']

    test = make.make_classifier_raw_data(logits_trigger, (pred_model1 == gt_model1), np.copy(model1_dict['test']['id']))

    classifier_trigger_dict = make.make_classifier_dict("trigger_classifier", "cifar10", train, test, performance)
    io.save_pickle('../../Definitions/Classifiers/tmp/trigger_random_threshold', classifier_trigger_dict)
    classifier = make.make_classifier("trigger_classifier", "../../Definitions/Classifiers/tmp/trigger_random_threshold")
    return classifier


if __name__ == "__main__":

    Classifier_Path = "../../Definitions/Classifiers/front45_models/"
    R = {}

    # Creating system
    sys = sb.SystemBuilder(verbose=False)
    bigClassifier = make.make_empty_classifier("big")
    sys.add_classifier(bigClassifier)
    trigger = make.make_trigger("probability_threshold_trigger", make.make_empty_classifier(), ["big"])
    sys.add_trigger(trigger)
    smallClassifier = make.make_empty_classifier("small", "probability_threshold_trigger")
    sys.add_classifier(smallClassifier)

    # Selecting models
    import os
    model_names = [f for f in os.listdir(Classifier_Path) if ".pkl" in f]
    models = [io.read_pickle(Classifier_Path+name) for name in model_names]
    for i, model in enumerate(models): model['name'] = model_names[i]
    models_sorted = sorted(models, key=lambda item: item['metrics']['params'])

    pairs = 6
    indices = list(range(0, len(models_sorted), int(len(models_sorted)/((pairs-1)*2))))

    classifiers_simple = [Classifier_Path+models_sorted[i]['name'] for i in indices[:pairs] + indices[:pairs] + indices[:-1:2]]
    classifiers_complex = [Classifier_Path+models_sorted[i]['name'] for i in list(reversed(indices[pairs:])) + indices[pairs:]+ indices[1::2]]

    for i, c_simple_file in enumerate(classifiers_simple):

        c_complex_file = classifiers_complex[i]
        print(c_simple_file, c_complex_file)

        r = {
            'max_class': [],
            'random_chain': [],
            'random_tree': [],
        }

        for th in np.arange(0, 1.01, 0.01):
            print(th)

            # Simple Classifier
            simple = make.make_classifier("small", c_simple_file, trigger.id)
            sys.replace("small", simple)
            simple_dict = io.read_pickle(c_simple_file)

            # Complex Classifier
            complex = make.make_classifier("big", c_complex_file)
            sys.replace("big", complex)

            """
            # Trigger 2 max class
            classifier_trigger = build_train_trigger(simple_dict, th)
            trigger = make.make_trigger("probability_threshold_trigger", classifier_trigger, ["big"], model="probability")
            trigger.id = "probability_threshold_trigger"
            sys.replace(trigger.id, trigger)

            results = eval.evaluate(sys, "small", check_classifiers=False).test
            r['2max_class'].append([results])
            """

            # Trigger max class
            classifier_trigger = build_train_trigger2(simple_dict, th)
            trigger = make.make_trigger("probability_threshold_trigger", classifier_trigger, ["big"], model="probability")
            sys.replace(trigger.id, trigger)

            results = eval.evaluate(sys, "small", check_classifiers=False).test
            r['max_class'].append([results])

            # Random trigger chain structure
            random_times_chain = []
            for i in range(5):
                classifier_trigger = build_train_trigger3(simple_dict, th)
                trigger = make.make_trigger("probability_threshold_trigger", classifier_trigger, ["big"])
                sys.replace(trigger.id, trigger)

                results = eval.evaluate(sys, "small", check_classifiers=False).test
                random_times_chain.append(results)

            r['random_chain'].append(random_times_chain)

            # Random trigger tree structure
            simple = make.make_classifier("small", c_simple_file)
            sys.replace("small", simple)
            random_times_tree = []
            for i in range(5):
                classifier_trigger = build_train_trigger3(simple_dict, th)
                trigger = make.make_trigger("probability_threshold_trigger", classifier_trigger, ["big", "small"])
                sys.replace(trigger.id, trigger)
                results = eval.evaluate(sys, "probability_threshold_trigger", check_classifiers=False).test
                random_times_tree.append(results)

            r['random_tree'].append(random_times_tree)

        R["system_"+c_simple_file+"_"+c_complex_file] = r

    io.save_pickle("./results/R.pkl", R)


