import Source.system_builder as sb
import Source.system_evaluator as ev
import Source.io_util as io
import Source.make_util as make
import Source.FastComposedModels_pb2 as fcm
import numpy as np
import os, sys, argparse

# Builds the skeleton of the ensemble
def build_skeleton(sys, data_location):
    # Merged classifier 0
    c00 = make.make_empty_classifier("c00")
    sys.add_classifier(c00)
    # Merged classifier 1
    c01 = make.make_empty_classifier("c01")
    sys.add_classifier(c01)
    # Merger c00 c01 classifiers (Average)
    merger = make.make_merger("merger",["c00","c01"], merge_type=fcm.Merger.AVERAGE) # Connectar al Trigger
    sys.add_merger(merger)
    # Trigger
    trigger = make.make_empty_trigger("trigger", ["c1"])
    sys.add_trigger(trigger)
    # Trigger data
    if not os.path.exists(data_location):
        os.makedirs(data_location)
    source = make.make_source(os.path.join(data_location,'train'), os.path.join(data_location,'test'), 2)
    data_trigger = make.make_data("trigger_data", 5e4, 1e4, source=source)
    sys.add_data(data_trigger)
    # Big classifier
    c1 = make.make_empty_classifier("c1")
    sys.add_classifier(c1)


# Returns a matrix (nxm) with prediction for all models (m)
def get_predictions(models):
    return 0


# Update dataset for train the trigger again
def update_dataset_trigger(models, n, train_path, test_path):
    return 0


# Returns the index of the model with accuracy > acc
def get_index_higher_accuracy(R_models, acc):
    return 0


def argument_parse(argv):
    parser = argparse.ArgumentParser(usage="vc.py [options]",
                                     description="Solution space by the voting trigger (3 models).")
    parser.add_argument("--datasets", nargs="*", default="*", help="Datasets to evaluate d1 d2 d3, default=*")
    parser.add_argument("--models_vote", type=int, default="2", help="Number of models that vote, default=2")
    parser.add_argument("--n", type=int, default="2", help="Number of models that need to agree in the vote, default=2")
    return parser.parse_args(argv)


# Start of script
if __name__ == "__main__":

    args = argument_parse(sys.argv[1:])
    pid = str(os.getpid())

    ##########################################################
    datasets_path = os.path.join(os.environ['FCM'], 'Definitions/Classifiers')
    datasets = args.datasets
    models_vote = args.models_vote
    n = args.n
    assert models_vote < 3 and n < 3, "Models cannot be > 2, not implemented yet"
    if datasets == "*":
        print("Evaluating all datasets available")
        datasets = [f for f in os.listdir(datasets_path) if 'tmp' not in f]
    ###########################################################

    for dataset in datasets:
        #########################################################################
        classifier_path = os.path.join(datasets_path, dataset)
        classifiers_name = [os.path.join(classifier_path, f) for f in os.listdir(classifier_path) if ".pkl" in f]
        out_dir = os.path.join("./results/", dataset)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        #########################################################################

        import Examples.paretto_front as paretto
        R_classifiers = {}
        for c_name in classifiers_name:
            sys = sb.SystemBuilder(verbose=False)
            c = make.make_classifier("classifier", c_name)
            sys.add_classifier(c)
            R_classifiers[c_name] = ev.evaluate(sys, c.id).test

        io.save_pickle(os.path.join(out_dir, "classifiers"), R_classifiers)
        classifiers_sorted = paretto.sort_results_by_accuracy(R_classifiers)
        classifiers_sorted_name = [v[0] for k in classifiers_sorted]

        # ENSEMBLE SKELETON
        sys = sb.SystemBuilder(verbose=False)
        data_location = os.path.join(os.environ['FCM'], 'Data', 'voting_trigger', str(os.getpid()))
        train_data = os.path.join(data_location, "train")
        test_data = os.path.join(data_location, "test")
        build_skeleton(sys, data_location)

        import itertools
        comb_two_cls = list(itertools.combinations(classifiers_name, models_vote))
        for comb in comb_two_cls:
            c00 = make.make_classifier("c00", comb[0])
            sys.replace("c00", c00)
            c01 = make.make_classifier("c01", comb[1])
            sys.replace("c01", c01)
            update_dataset_trigger(["c00", "c01"], n, train_data, test_data)
            acc1 = ev.evaluate(sys, "c00").test['c00'].accuracy
            acc2 = ev.evaluate(sys, "c01").test['c01'].accuracy
            acc = max(acc1, acc2)
            i = get_index_higher_accuracy(classifiers_sorted, acc)
            for big_classifier in classifiers_sorted_name[i:]:
                c1 = make.make_classifier("c1", big_classifier)
                sys.replace("c1", c1)



