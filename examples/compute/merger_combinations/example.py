import numpy as np
import source.system_builder as sb
import source.make_util as make
import source.FastComposedModels_pb2 as fcm
import source.system_evaluator as eval
import source.io_util as io
from threading import Thread, Lock
import queue
import itertools
import argparse

results = {}

def create_evaluate_system(work, results):
    # Creating system
    lock = Lock()
    while True:
        protocol, subset = work.get()
        sys = sb.SystemBuilder(verbose=False)
        classifiers_ids = []
        for m in subset:
            file = m
            model = make.make_classifier(m, file)
            sys.add_classifier(model)
            classifiers_ids.append(model.id)

        merger = make.make_merger("MERGER", classifiers_ids, merge_type=protocol)
        sys.add_merger(merger)
        r = eval.evaluate(sys, merger.id)
        # if results: results = io.read_pickle('./results/R_'+str(protocol)+'_'+str(n_models))
        results['system_' + '_'.join(classifiers_ids) + '_protocol' + str(protocol)] = r

        work.task_done()


def sort_models_params(models):
    size = [c_dict['metrics']['params'] for c_dict in [io.read_pickle(model) for model in models]]
    position = np.argsort(size)
    models_sorted = ["" for x in range(len(models))]
    for i in range(len(models)):
        models_sorted[position[i]] = models[i]
    return models_sorted


def argument_parse(argv):
    import argparse
    parser = argparse.ArgumentParser(usage="example.py [options]",
                                     description="Applies ensamble techniques to combination of models.")
    parser.add_argument("--datasets", nargs="*", default="*", help="Datasets to evaluate  d1 d2 d3 ")
    parser.add_argument("--n_models", nargs="*", type=int, default=[3, 5], help="Perform all combinations of n_models with different ensamble techniques")
    parser.add_argument("--merge_protocol", nargs="*", type=int, default=list(range(9)), help="Merge protocol: 1,2,3,4,5,6,7,8")
    parser.add_argument("--threads", default=4, type=int, help="Code can be concurrent, how many threads computing the combinations of models")
    parser.add_argument("--step", default=2, type=int, help="Higher the number, less combinations of models")

    return parser.parse_args(argv)


if __name__ == "__main__":
    import os
    import sys

    args = argument_parse(sys.argv[1:])  # Parameters for the control of the program

    ##########################################################
    datasets = args.datasets
    if datasets == "*":
        print("Evaluating all datasets available")
        datasets = [f for f in os.listdir("../../definitions/Classifiers/") if 'tmp' not in f]
    step = args.step
    n_threads = args.threads
    merge_protocols = args.merge_protocol
    num_merged_models = args.n_models
    ###########################################################

    results = {}

    for dataset in datasets:
        print("DATASAET: ", dataset)

        ##################################################################
        Classifier_Path = "../../definitions/Classifiers/" + dataset + "/"
        out_dir = os.path.join('./results', dataset) + '/'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        ##################################################################

        models = [Classifier_Path+f for f in os.listdir(Classifier_Path) if ".pkl" in f]
        models_sorted = sort_models_params(models)

        for n_models in num_merged_models:
            print("N_MODELS:", n_models)
            for protocol in merge_protocols:
                print("MERGE:", protocol)
                for i in range(step+1 if n_models == 5 else step):
                    print("STEP:", i)
                    results.clear()
                    models = models_sorted[i::step]
                    model_subsets = itertools.combinations(models, n_models)
                    for subset in model_subsets:
                        print(subset)

                        sys = sb.SystemBuilder(verbose=False)
                        classifiers_ids = []
                        for m in subset:
                            file = m
                            model = make.make_classifier(m, file)
                            sys.add_classifier(model)
                            classifiers_ids.append(model.id)

                        merger = make.make_merger("MERGER", classifiers_ids, merge_type=protocol)
                        sys.add_merger(merger)
                        r = eval.evaluate(sys, merger.id)
                        results['_'.join(classifiers_ids) + '_protocol' + str(protocol)] = r

                    io.save_pickle(os.path.join(out_dir, 'R_'+str(protocol)+'_'+str(n_models)+'_part'+str(i)), results)