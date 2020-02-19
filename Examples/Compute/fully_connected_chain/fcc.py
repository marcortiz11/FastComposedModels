import numpy as np
import Source.system_builder as sb
import Source.make_util as make
import Source.io_util as io
import Source.system_evaluator as eval
import Examples.metadata_manager_results as manager_results
import os, random


def update_dataset(model_file, ths, train_path, test_path):
    model = io.read_pickle(model_file)
    # Test
    dataset_test = model['test']
    dataset_test['ths'] = ths
    io.save_pickle(test_path, dataset_test)
    # Train
    dataset_train = model['train']
    dataset_train['ths'] = ths
    io.save_pickle(train_path, dataset_train)


def argument_parse(argv):
    import argparse
    parser = argparse.ArgumentParser(usage="fcc.py [options]",
                                     description="This code experiments the performance of fully connected "
                                                 "chain of 3 models vs simple chain of 3 models by exploring all "
                                                 "combinations.")
    parser.add_argument("--datasets", nargs="*", default="*", help="Datasets to evaluate d1 d2 d3, default=*")
    parser.add_argument("--step_th", default=0.1, type=float, help="Threshold step, default=0.1")
    parser.add_argument("--fc", default=0, type=int, help="All combinations with fully connected chain, default=fc")
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
    fc = args.fc >= 1
    pid = str(os.getpid())
    ###########################################################

    for dataset in datasets:

        #########################################################################
        Classifier_Path = os.environ['FCM']+"/Definitions/Classifiers/" + dataset + "/"
        models = [Classifier_Path + f for f in os.listdir(Classifier_Path) if ".pkl" in f]
        out_dir = os.path.join("./results/", dataset)
        data_path = os.environ['FCM']+"/Data/"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        #########################################################################

        import Examples.paretto_front as paretto
        R_models = {}
        for model in models:
            sys = sb.SystemBuilder(verbose=False)
            c = make.make_classifier("classifier", model)
            sys.add_classifier(c)
            R_models[model] = eval.evaluate(sys, c.id, phases=["test"])

        io.save_pickle(os.path.join(out_dir, "models"), R_models)
        front = paretto.get_front_time_accuracy(R_models, phase="test")
        front_sorted = paretto.sort_results_by_accuracy(front, phase="test")


        # ENSEMBLE SKELETON
        sys = sb.SystemBuilder(verbose=False)
        trigger0_train_dataset = os.path.join(data_path, "train_trigger0"+pid)
        trigger0_test_dataset = os.path.join(data_path, "test_trigger0"+pid)
        trigger1_train_dataset = os.path.join(data_path, "train_trigger1"+pid)
        trigger1_test_dataset = os.path.join(data_path, "test_trigger1"+pid)
        # Classifier 0
        c0 = make.make_empty_classifier("c0", component_id="trigger0"+pid)
        sys.add_classifier(c0)
        # Data for trigger 0
        source = make.make_source(trigger0_train_dataset, trigger0_test_dataset, 2)
        data0 = make.make_data("trigger0_data", int(5e4), int(1e4), source=source)
        sys.add_data(data0)
        # Trigger 0
        trigger0 = make.make_trigger("trigger0"+pid, make.make_empty_classifier(data_id="trigger0_data"),
                                     ["c2", "c1"], model="probability")
        sys.add_trigger(trigger0)
        # Classifier 1
        c2 = make.make_empty_classifier("c2")
        sys.add_classifier(c2)
        # Data for trigger 1
        source = make.make_source(trigger1_train_dataset, trigger1_test_dataset, 2)
        data1 = make.make_data("trigger1_data", int(5e4), int(1e4), source=source)
        sys.add_data(data1)
        # Trigger 1
        trigger1 = make.make_trigger("trigger1"+pid, make.make_empty_classifier(data_id="trigger1_data"),
                                     ["c2"], model="probability")
        sys.add_trigger(trigger1)
        # Classifier 2
        c1 = make.make_empty_classifier("c1", component_id="trigger1"+pid)
        sys.add_classifier(c1)

        # Classifier 0
        records = {}

        for ic0 in range(len(models)):
            c0 = models[ic0]
            sys.replace("c0", make.make_classifier("c0", c0, "trigger0"+pid))

            # Thresholds of Trigger 0
            for th0 in np.arange(0.2, 1, step_th):
                for th1 in np.arange((th0+step_th)*fc, fc + step_th, step_th):
                    update_dataset(c0, [th0, th1], trigger0_train_dataset, trigger0_test_dataset)
                    trigger = make.make_trigger("trigger0"+pid, make.make_empty_classifier(data_id="trigger0_data"),
                                                ["c2", "c1"], model="probability_multiple_classifiers")
                    sys.replace("trigger0"+pid, trigger)

                    # Classifier 1
                    for ic1 in range(ic0+1, len(models)):
                        c1 = models[ic1]
                        sys.replace("c1", make.make_classifier("c1", c1, "trigger1"+pid))

                        # Threshold for Trigger 1
                        for th2 in np.arange(0.2, 1, step_th):
                            update_dataset(c1, [th1], trigger1_train_dataset, trigger1_test_dataset)
                            trigger = make.make_trigger("trigger1"+pid, make.make_empty_classifier(data_id="trigger1_data"),
                                                        ["c2"], model="probability_multiple_classifiers")
                            sys.replace("trigger1"+pid, trigger)

                            # Classifier 2
                            for ic2 in range(ic1+1, len(models)):
                                c2 = models[ic2]
                                sys.replace("c2", make.make_classifier("c2", c2))

                                # Evaluate
                                id = "%s_(%f,%f)_%s_(%f)_%s" % (c0.split('/')[-1], th0, th1, c1.split('/')[-1], th2, c2.split('/')[-1])
                                results = eval.evaluate(sys, "c0")
                                import sys as sy
                                records[id] = results
                                print(id)

                        # Buidar solucions que no pertanyen al paretto front
                        optimal_chains = paretto.get_front_time_accuracy(records, phase="test")
                        records.clear()
                        records = optimal_chains
                        print("%d chains in the front" % len(records.keys()))


        # Crear la meta_data
        meta_data_file = os.path.join(os.environ['FCM'],
                                      'Examples',
                                      'Compute',
                                      'fully_connected_chain',
                                      'results',
                                      'metadata.json')
        id = str(random.randint(0, 1e8))
        results_loc = os.path.join('./results', dataset, id)
        comments = "ONLY PARETTO FRONT SAVED"
        meta_data_result = manager_results.metadata_template(id, dataset, results_loc, comments)

        # Obtenir el diccionari de params
        params = args.__dict__

        # Guardar els resultats en la carpeta del dataset
        manager_results.save_results(meta_data_file, meta_data_result, params, records)

        records = {}