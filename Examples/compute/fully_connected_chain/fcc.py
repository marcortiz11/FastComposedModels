import numpy as np
import Source.protobuf.system_builder_serializable as sb
import Source.protobuf.make_util as make
import Source.io_util as io
import Source.system_evaluator as eval
import Examples.metadata_manager_results as manager_results
from Source.genetic_algorithm.operations_mutation import extend_merged_chain
from Source.system_evaluator_utils import pretty_print
import os, random


def build_evaluate_chain(files: [str], ths: [float]):
    assert len(files) > 0 and len(files) == len(ths)+1

    sys = sb.SystemBuilder(verbose=False)
    classifier = make.make_classifier(os.path.basename(files[0]), files[0])
    sys.add_classifier(classifier)
    sys.set_start(classifier.id)

    # Automatically build the chain with written mutation operations
    for i, file in enumerate(files[:-1]):
        extend_merged_chain(sys, os.path.basename(file), os.path.basename(files[i+1]), ths[i], files[i+1])

    result = eval.evaluate(sys)
    return result


def argument_parse(argv):
    import argparse
    parser = argparse.ArgumentParser(usage="fcc.py [options]",
                                     description="This code experiments the performance of fully connected "
                                                 "chain of 3 models vs simple chain of 3 models by exploring all "
                                                 "combinations.")
    parser.add_argument("--dataset",  default="sota_models_caltech256-32-dev_validation")
    parser.add_argument("--step_th", default=0.1, type=float, help="Threshold step, default=0.1")
    parser.add_argument("--fc", default=0, type=int, help="All combinations with fully connected chain")
    parser.add_argument("--parallel", default=1, type=int, help="Number of cores paralellism")
    parser.add_argument("--comment", default="", type=str, help="Comments about the experiment")
    return parser.parse_args(argv)


if __name__ == "__main__":

    import sys
    args = argument_parse(sys.argv[1:])
    os.environ['TMP'] = 'Definitions/Classifiers/tmp/'+args.dataset
    if not os.path.exists(os.path.join(os.environ['FCM'], os.environ['TMP'])):
        os.makedirs(os.path.join(os.environ['FCM'], os.environ['TMP']))

    ##########################################################
    datasets = [args.dataset]
    step_th = args.step_th
    fc = args.fc >= 1
    pid = str(os.getpid())
    ##########################################################

    for dataset in datasets:

        #########################################################################
        Classifier_Path = os.environ['FCM']+"/Definitions/Classifiers/" + dataset + "/"
        model_paths = [Classifier_Path + f for f in os.listdir(Classifier_Path) if ".pkl" in f]
        out_dir = os.path.join("./results/", dataset)
        data_path = os.environ['FCM']+"/Datasets/"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        #########################################################################

        import Examples.study.paretto_front as paretto
        R_models = {}
        for mpath in model_paths:
            sys = sb.SystemBuilder(verbose=False)
            classifier_id = os.path.basename(mpath)
            c = make.make_classifier(classifier_id, mpath)
            sys.add_classifier(c)
            sys.set_start(classifier_id)
            R_models[classifier_id] = eval.evaluate(sys, phases=["test", "val"])

        #front = paretto.get_front_time_accuracy(R_models, phase="test")
        #front_sorted = paretto.sort_results_by_accuracy(front, phase="test")
        models_paths = [Classifier_Path + k for k, v in paretto.sort_results_by_accuracy(R_models, phase="val")]
        records = R_models

        # Combinations
        for ic0 in range(len(models_paths)):
            c0 = models_paths[ic0]
            for th0 in np.arange(0, 1+step_th, step_th):
                for ic1 in range(ic0+1, len(models_paths)):
                    c1 = models_paths[ic1]
                    #for th2 in np.arange(0.2, 1, step_th):
                    #   for ic2 in range(ic1+1, len(models)):
                    #       c2 = models[ic2]

                    r = build_evaluate_chain([c0, c1], [th0])
                    sysid = "%s-%f-%s" % (c0.split('/')[-1], th0, c1.split('/')[-1])
                    print(sysid)
                    records[sysid] = r

        # Crear la meta_data
        meta_data_file = os.path.join(os.environ['FCM'],
                                      'Examples',
                                      'compute',
                                      'fully_connected_chain',
                                      'results',
                                      'metadata.json')
        id = str(random.randint(0, 1e16))
        results_loc = os.path.join('Examples/compute/fully_connected_chain/results', dataset, id)
        meta_data_result = manager_results.metadata_template(id, dataset, results_loc, args.comment)

        # Obtenir el diccionari de params
        params = args.__dict__

        # Guardar els resultats en la carpeta del dataset
        manager_results.save_results(meta_data_file, meta_data_result, params, records)

        records = {}
