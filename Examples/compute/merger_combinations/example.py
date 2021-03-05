import numpy as np
import Source.protobuf.system_builder_serializable as sb
import Source.protobuf.make_util as make
import Source.system_evaluator as eval
import Source.io_util as io
from threading import Thread, Lock
import itertools, random
from Examples.compute.chain_genetic_algorithm.utils import generate_system_id

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


def evaluate_single_models(models, results):
    for m in models:
        sys = sb.SystemBuilder()
        c = make.make_classifier(m, m)
        sys.add_classifier(c)
        sys.set_start(m)
        results[generate_system_id(sys)] = eval.evaluate(sys, sys.get_start(), phases=["test"])


def argument_parse(argv):
    import argparse
    parser = argparse.ArgumentParser(usage="example.py [options]",
                                     description="Applies ensamble techniques to combination of models.")
    parser.add_argument("--datasets", nargs="*", default="*", help="Datasets to evaluate  d1 d2 d3 ")
    parser.add_argument("--n_models", nargs="*", type=int, default=[3], help="Perform all combinations of n_models with different ensamble techniques")
    parser.add_argument("--merge_protocol", nargs="*", type=int, default=list(range(9)), help="Merge protocol: 1,2,3,4,5,6,7,8")
    parser.add_argument("--threads", default=4, type=int, help="Code can be concurrent, how many threads computing the combinations of models")
    parser.add_argument("--step", default=1, type=int, help="Higher the number, less combinations of models")

    return parser.parse_args(argv)


if __name__ == "__main__":
    import os
    import sys

    args = argument_parse(sys.argv[1:])  # Parameters for the control of the program

    ##########################################################
    datasets = args.datasets
    datasets_path = os.path.join(os.environ['FCM'], 'Definitions/Classifiers')
    if datasets == "*":
        print("Evaluating all datasets available")
        datasets = [f for f in os.listdir(datasets_path) if 'tmp' not in f]
    step = args.step
    n_threads = args.threads
    merge_protocols = args.merge_protocol
    num_merged_models = args.n_models
    ###########################################################

    results = {}

    for dataset in datasets:
        print("DATASAET: ", dataset)

        ##################################################################
        classifiers_path = os.path.join(datasets_path, dataset)
        out_dir = os.path.join('./results', dataset)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        ##################################################################

        models = [os.path.join(classifiers_path, f) for f in os.listdir(classifiers_path) if ".pkl" in f]
        evaluate_single_models(models, results)

        for n_models in num_merged_models:
            print("N_MODELS:", n_models)
            for protocol in merge_protocols:
                print("MERGE:", protocol)
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
                    sys.set_start(merger.id)
                    r = eval.evaluate(sys, sys.get_start(), phases=["test"])
                    results[generate_system_id(sys)] = r

                # Save the evaluation results
                import Examples.metadata_manager_results as manager_results

                meta_data_file = os.path.join(os.environ['FCM'],
                                              'Examples',
                                              'compute',
                                              'merger_combinations',
                                              'results',
                                              'metadata.json')

                id = str(random.randint(0, 1e16))
                results_loc = os.path.join('Examples/compute/merger_combinations/results', dataset, id)
                meta_data_result = manager_results.metadata_template(id, dataset, results_loc, "")

                # Save the ensemble evaluation results
                params = args.__dict__
                manager_results.save_results(meta_data_file, meta_data_result, params, results)


