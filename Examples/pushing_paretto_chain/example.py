import numpy as np
import Source.system_builder as sb
import Source.make_util as make
import Source.io_util as io
import Examples.paretto_front as paretto
import Source.system_evaluator as eval
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import Examples.plot as myplt


def update_dataset(model_file, th, train_path, test_path, val_path):
    model = io.read_pickle(model_file)
    # Test
    dataset_test = model['test']
    dataset_test['th'] = th
    io.save_pickle(test_path, dataset_test)
    # Train
    dataset_train = model['train']
    dataset_train['th'] = th
    io.save_pickle(train_path, dataset_train)
    # Validation
    dataset_val = model['val']
    dataset_val['th'] = th
    io.save_pickle(val_path, dataset_val)



def plot_all_iteration(iter, dir, color="green"):
    plt.figure(0)
    plt.title("")
    plt.xlabel("Time (s)")
    plt.ylabel("Accuracy")
    plt.xscale("log")

    onlyfiles = [join(dir, f) for f in listdir(dir) if isfile(join(dir, f)) and f.split(";")[0] == str(iter)]

    for file in onlyfiles:
        R = io.read_pickle(file)
        myplt.plot_accuracy_time(R, system_color=color)


# Get ids sorted from the optimal front of the iteration
def get_optimal_front_iteration_ids(iter, dir, front_opt):
    front = {}
    models = [join(dir, f) for f in listdir(dir) if isfile(join(dir, f)) and f.split(";")[0] == str(iter)]
    for m in models:
        r = io.read_pickle(m)
        front.update(r)
        if front_opt == "time_acc":
            front = paretto.get_front_time_accuracy(front, phase="val")
        else:
            front = paretto.get_front_params_accuracy(front, phase="val")
    sorted_models = paretto.sort_results_by_accuracy(front, phase="val")
    return sorted_models


def argument_parse(argv):
    import argparse
    parser = argparse.ArgumentParser(usage="example.py [options]",
                                     description="Applies ensamble techniques to combination of models.")
    parser.add_argument("--datasets", nargs="*", default="*", help="Datasets to evaluate d1 d2 d3")
    parser.add_argument("--front", default="time_acc", help="Which front to push: time_acc or params_acc")
    parser.add_argument("--step", default=0.1, type=float, help="Threshold step, default=0.1")
    parser.add_argument("--iterations", default=2, type=int, help="Number of iterations in recursive creation of classifiers")
    return parser.parse_args(argv)


if __name__ == "__main__":

    import os
    import sys

    args = argument_parse(sys.argv[1:])

    ##########################################################
    datasets = args.datasets
    if datasets == "*":
        print("Evaluating all datasets available")
        datasets = [f for f in os.listdir("../../Definitions/Classifiers/") if 'tmp' not in f]
    step = args.step
    front = args.front
    iterations = args.iterations
    ###########################################################

    for dataset in datasets:

        print("Evaluating dataset %s" % dataset)

        ##################################################################
        Classifier_Path = os.path.join("../../Definitions/Classifiers/", dataset)
        out_dir = os.path.join("./results/", dataset)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        ##################################################################

        models = [os.path.join(Classifier_Path, f) for f in os.listdir(Classifier_Path) if ".pkl" in f]

        R = {}
        combined_model_id = 0

        tmp_classifiers_dir = "../../Definitions/Classifiers/tmp/"

        R_models = {}
        for model in models:
            sys = sb.SystemBuilder(verbose=False)
            c = make.make_classifier("classifier", model)
            sys.add_classifier(c)
            R_models[model] = eval.evaluate(sys, c.id, phases=["val", "test"])

        io.save_pickle(os.path.join(out_dir, "models"), R_models)
        if front == "time_acc":
            front_models = paretto.get_front_time_accuracy(R_models, phase="val")
        elif front == "params_acc":
            front_models = paretto.get_front_params_accuracy(R_models, phase="val")
        else:
            raise ValueError("Choose a front to optimize: params_acc or time_acc")

        front_sorted = paretto.sort_results_by_accuracy(front_models, phase="val")
        io.save_pickle(os.path.join(out_dir, "front0"), front_models)
        front_sorted_ids = [val[0] for val in front_sorted]
        print(front_sorted_ids)

        # Building system skeleton
        sys = sb.SystemBuilder(verbose=False)
        bigClassifier = make.make_empty_classifier("big")
        sys.add_classifier(bigClassifier)
        train_path='../../Data/train_trigger_threshold.pkl'
        test_path = '../../Data/test_trigger_threshold.pkl'
        val_path = '../..//Data/val_trigger_threshold.pkl'
        source = make.make_source(train_path, test_path, 2, val_path)
        data = make.make_data("trigger_data", int(5e4), int(1e4), source=source)
        sys.add_data(data)
        trigger = make.make_trigger("threshold", make.make_empty_classifier(), ["big"])
        sys.add_trigger(trigger)
        smallClassifier = make.make_empty_classifier("small", "threshold")
        sys.add_classifier(smallClassifier)

        for iteration in range(iterations):

            print("Iteration:", iteration+1)

            # Model 1
            for mi, m in enumerate(front_sorted_ids):

                R.clear()

                smallClassifier = make.make_classifier("small", m, "threshold")
                smallClassifier_dict = io.read_pickle(m)
                sys.replace("small", smallClassifier)

                # For different thresholds
                for th in np.arange(0+step, 1, step):
                    update_dataset(m, th, train_path, test_path, val_path)
                    trigger = make.make_trigger("threshold", make.make_empty_classifier(data_id="trigger_data"), ["big"], model="probability")
                    sys.replace(trigger.id, trigger)

                    # Model 2
                    for m_ in front_sorted_ids[mi+1:]:

                        print(m, th, m_)

                        bigClassifier = make.make_classifier("big", m_)
                        sys.replace("big", bigClassifier)

                        filename = os.path.join(tmp_classifiers_dir, str(combined_model_id)) + '_' + ''.join(dataset) + '.pkl'

                        classifier_chain, eval_results_classifier_chain = sys.build_classifier_dict(filename, "small", phases= ["val", "test"])
                        R[filename] = eval_results_classifier_chain
                        R[filename].test['th'] = th
                        R[filename].test['small_model'] = m
                        R[filename].test['big_model'] = m_

                        if "system" in m_:
                            if "system" in m:
                                R[filename].test['num'] = 4
                            else:
                                R[filename].test['num'] = 3
                        elif "system" in m:
                            R[filename].test['num'] = 3
                        else:
                            R[filename].test['num'] = 2

                        """
                        # Only save classifiers that currently are in the front
                        if front == "time_acc":
                            temporal_front = paretto.get_front_time_accuracy(R)
                        else:
                            temporal_front = paretto.get_front_params_accuracy(R)
                        if filename in temporal_front:
                            
                        """
                        io.save_pickle(filename, classifier_chain)

                        combined_model_id += 1

                io.save_pickle(os.path.join(out_dir, str(iteration+1) + ";" + str(mi)), R)

            front_iteration_sorted = get_optimal_front_iteration_ids(iteration+1, out_dir, front)
            front_models.update(front_iteration_sorted)
            if front == "time_acc":
                front_models = paretto.get_front_time_accuracy(front_models, phase="val")
            else:
                front_models = paretto.get_front_params_accuracy(front_models, phase="val")
            front_sorted = paretto.sort_results_by_accuracy(front_models, phase="val")
            front_sorted_ids = [item[0] for item in front_sorted]

            io.save_pickle(os.path.join(out_dir, "front"+str(iteration+1)), front_models)

