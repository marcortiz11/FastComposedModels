import Examples.study.paretto_front as front
import Examples.metadata_manager_results as results_manager
import Source.genetic_algorithm.fitting_functions as fit_fun
from Source.system_evaluator_utils import pretty_print
import Source.io_util as io
import statistics as stats
import numpy as np
import os
import sys


def print_metrics_model(id, r):
    print()
    print(id)
    print("\t Test accuracy: %f" % r['system'].accuracy)
    for classifier in r:
        if 'trigger' not in classifier:
            print("\t\t Test accuracy %s: %f" % (classifier, r[classifier].accuracy))

    print("\t Model parameters: %f" % (r['system'].params/1e6))
    for classifier in r:
        if 'trigger' not in classifier:
            print("\t\t Model parameters %s: %f * 1e6" % (classifier, r[classifier].params/1e6))

    print("\t Instances processed: %d" % (r['system'].instances))
    for classifier in r:
        if 'trigger' not in classifier:
            print("\t\tInstances processed %s: %d" % (classifier, r[classifier].instances))

    print("\t Dataset Evaluation time: %f s" % (r['system'].time))


if __name__ == "__main__":

    experiment = 'bagging_boosting_of_chains_GA'
    query_params = {
        'dataset': "sota_models_cifar10-32-dev_validation",
        'selection': 'nfit',
        'experiment': 'bagging_boosting_of_chains_GA_1',
        'iterations': 200,
        'pm': 0.9,
        'a':  [
                0.8,
                0.2,
                0,
            ],
        'k': 10,
        'population': 1000,
        'offspring': 50,
        'comment': 'Fitness_normalize_Crossover_v2'
    }
    num = 1
    phase = "test"

    # 1) G.A. Chain ensembles
    GA_results_metadata_file = os.path.join(os.environ['FCM'],
                                            'Examples',
                                            'compute',
                                            experiment,
                                            'results',
                                            'metadata.json')

    # Get evaluation results from query
    GA_res_loc = results_manager.get_results_by_params(GA_results_metadata_file, query_params)
    GA_res_loc = GA_res_loc[-num:]
    GA_res_loc = [os.path.join(path, 'results_ensembles.pkl') for path in GA_res_loc]

    # 2) Single Models
    models = dict([(k, r) for k, r in io.read_pickle(GA_res_loc[0]).items() if len(r.test) < 3])
    models_front = front.get_front_time_accuracy(models, phase=phase)
    sorted_models_front = front.sort_results_by_time(models_front, phase=phase)

    accurate_NN_result = models[sorted_models_front[-1][0]].test if phase == "test" \
                        else models[sorted_models_front[-1][0]].val
    acc = accurate_NN_result['system'].accuracy
    time = accurate_NN_result['system'].time
    params = accurate_NN_result['system'].params
    print_metrics_model(sorted_models_front[-1][0], accurate_NN_result)

    from Source.genetic_algorithm.fitting_functions import make_limits_dict, update_limit_dict
    limit = make_limits_dict()
    update_limit_dict(limit, models, phase=phase)

    # 3) Speedup and Parameter decrease
    speedup = []
    param_incrase = []
    acc_increase = []

    for res_loc in GA_res_loc:
        GA_chains = io.read_pickle(res_loc)
        list_chain_res = list(GA_chains.values())
        list_chain_keys = list(GA_chains.keys())
        list_fit_vals = np.array(fit_fun.f2_time_param_penalization(list_chain_res, query_params['a'], limit, phase))*-1

        sorted = np.argsort(list_fit_vals)
        fittest_model_id = list_chain_keys[sorted[0]]
        fittest_model_result = GA_chains[fittest_model_id].val if phase == "val" else GA_chains[fittest_model_id].test

        """
        for id in sorted:
            speedup_chain_id = list_chain_keys[id]
            speedup_chain_result = GA_chains[speedup_chain_id]
            if True:
                break
        """

        print_metrics_model(fittest_model_id, fittest_model_result)

        # Compute increase in params
        # pretty_print(fittest_model_result)
        speedup.append(time/fittest_model_result['system'].time)
        param_incrase.append(fittest_model_result['system'].params/params)
        acc_increase.append(fittest_model_result['system'].accuracy-acc)

    if len(GA_res_loc) > 0:
        print("\nImprovements:")
        print("\tAvg speedup:", stats.mean(speedup))
        print("\tAvg param incrase:", stats.mean(param_incrase))
        print("\tAvg acc increase:", stats.mean(acc_increase))
    else:
        print("No data available")
