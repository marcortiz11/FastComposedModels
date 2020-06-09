import numpy as np
import Source.system_builder as sb
import Source.make_util as make
import Source.io_util as io
import Source.system_evaluator as eval
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from threading import Thread, Lock
import time, queue
import os


def update_dataset(model_file, th, train_path, test_path):
    model = io.read_pickle(model_file)
    # Test
    dataset_test = model['test']
    dataset_test['th'] = th
    io.save_pickle(test_path, dataset_test)
    # Train
    dataset_train = model['train']
    dataset_train['th'] = th
    io.save_pickle(train_path, dataset_train)


def evaluate(work, records, tid):
    lock = Lock()
    while True:

        m, th, m_ = work.get()
        print(m, th, m_)

        small_dict = io.read_pickle(m)
        test_images = len(small_dict['test']['gt'])
        train_images = len(small_dict['train']['gt'])
        data_path = "../../Data/"
        trigger_train_dataset = os.path.join(data_path, "train_trigger_" + str(tid))
        test_train_dataset = os.path.join(data_path, "test_trigger_" + str(tid))

        sys = sb.SystemBuilder(verbose=False)

        # Complex classifier
        bigClassifier = make.make_classifier("big", m_)
        sys.add_classifier(bigClassifier)

        # Data
        source = make.make_source(trigger_train_dataset, test_train_dataset, 2)
        data = make.make_data("trigger_data", train_images, test_images, source=source)
        sys.add_data(data)
        update_dataset(m, th, trigger_train_dataset, test_train_dataset)

        # Trigger
        trigger = make.make_trigger("trigger", make.make_empty_classifier(data_id="trigger_data"),
                                    ["big"], model="probability")
        sys.add_trigger(trigger)

        # Simple classifier
        smallClassifier = make.make_classifier("small", m, "trigger")
        sys.add_classifier(smallClassifier)

        results = eval.evaluate(sys, "small", check_classifiers=False)
        records["system_" + m + ';' + m_ + ';' + str(th)] = results.test

        lock.acquire()
        if m_ not in records:  # Evaluate individual models in order to plot
            records[m_] = eval.evaluate(sys, 'big').test
        lock.release()

        work.task_done()


def argument_parse(argv):
    import argparse
    parser = argparse.ArgumentParser(usage="example.py [options]",
                                     description="Applies ensamble techniques to combination of models.")
    parser.add_argument("--datasets", nargs="*", default="*", help="Datasets to evaluate d1 d2 d3")
    parser.add_argument("--step_th", default=0.1, type=float, help="Threshold step, default=0.1")
    parser.add_argument("--step_models", default=1, type=int, help="Model selection step, default=2 (every two models select one)")
    parser.add_argument("--n_models", nargs="*", default=[2], help="Number of models in the chain")
    parser.add_argument("--threads", default=2, type=int, help="Number of threads solving the problem")
    return parser.parse_args(argv)


if __name__ == "__main__":

    import sys
    args = argument_parse(sys.argv[1:])

    ##########################################################
    datasets = args.datasets
    if datasets == "*":
        print("Evaluating all datasets available")
        datasets = [f for f in os.listdir("../../Definitions/Classifiers/") if 'tmp' not in f]
    step_th = args.step_th
    step_models = args.step_models
    n_threads = args.threads
    ###########################################################

    for dataset in datasets:

        # Results of the chain
        records = {}

        #########################################################################
        Classifier_Path = "../../Definitions/Classifiers/" + dataset + "/"
        models = [Classifier_Path + f for f in os.listdir(Classifier_Path) if ".pkl" in f]
        out_dir = os.path.join("./results/", dataset)
        data_path = "../../Data/"
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        #########################################################################
        import Examples.paretto_front as paretto

        R_models = {}
        for model in models:
            sys = sb.SystemBuilder(verbose=False)
            c = make.make_classifier("classifier", model)
            sys.add_classifier(c)
            R_models[model] = eval.evaluate(sys, c.id).test

        io.save_pickle(os.path.join(out_dir, "models"), R_models)

        front_sorted = paretto.sort_results_by_accuracy(R_models)
        models = [val[0] for val in front_sorted]
        print(models)

        sys = sb.SystemBuilder(verbose=False)
        trigger_train_dataset = os.path.join(data_path, "train_trigger_0")
        test_train_dataset = os.path.join(data_path, "test_trigger_0")


        # Model 1
        for im, m in enumerate(models):

            smallClassifier = make.make_classifier("small", m, "trigger")
            sys.replace("small", smallClassifier)

            small_dict = io.read_pickle(m)
            test_images = len(small_dict['test']['gt'])
            train_images = len(small_dict['train']['gt'])

            # For different thresholds
            for th in np.arange(0+step_th, 1, step_th):

                # Data
                source = make.make_source(trigger_train_dataset, test_train_dataset, 2)
                data = make.make_data("trigger_data", train_images, test_images, source=source)
                sys.replace("trigger_data", data)
                update_dataset(m, th, trigger_train_dataset, test_train_dataset)

                trigger = make.make_trigger("trigger", make.make_empty_classifier(data_id="trigger_data"),
                                            ["big"], model="probability")
                sys.replace("trigger", trigger)

                # Model 2
                for im_, m_ in enumerate(models[im+1:]):
                    print(m, th, m_)

                    bigClassifier = make.make_classifier("big", m_)
                    sys.replace("big", bigClassifier)

                    results = eval.evaluate(sys, "small", check_classifiers=False)
                    records[m + ';' + m_ + ';' + str(th)] = results

        io.save_pickle(os.path.join(out_dir, "ensembles"), records)
