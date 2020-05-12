import examples.paretto_front as front
import examples.metadata_manager_results as results_manager
import source.genetic_algorithm.fitting_functions as fit_fun
import source.io_util as io
import statistics as stats
import numpy as np
import os

if __name__ == "__main__":

    experiment = 'chain_genetic_algorithm'
    query_params = {
        'dataset': 'sota_models_svhn-32-dev_validation',
        'comment': 'Optimization_acc_realCPUtime_param',
        'selection': 'nfit',
        'a': 5,
        'iterations': 20
    }

    # 0) Single DNNs
    models = io.read_pickle(
        os.path.join(os.environ['FCM'], 'small_examples', 'models_evaluation', query_params['dataset'], 'models.pkl'))
    models_front = front.get_front_time_accuracy(models, phase="test")
    sorted_models_front = front.sort_results_by_time(models_front, phase="test")

    # Best NN test eval results
    accurate_NN_result = models[sorted_models_front[-1][0]]
    acc = accurate_NN_result.test['system'].accuracy
    time = accurate_NN_result.test['system'].time
    params = accurate_NN_result.test['system'].params
    print("Best NN: %s" % sorted_models_front[-1][0])

    # 1) G.A. Chain ensembles
    GA_results_metadata_file = os.path.join(os.environ['FCM'],
                                            'examples',
                                            'compute',
                                            experiment,
                                            'results',
                                            'metadata.json')
    # Get evaluation results from query
    GA_res_loc = results_manager.get_results_by_params(GA_results_metadata_file, query_params)
    print('\n'.join(GA_res_loc))

    speedup = []
    param_incrase = []

    for res_loc in GA_res_loc:
        GA_chains = io.read_pickle(res_loc)
        front_GA = front.get_front_time_accuracy(GA_chains, phase="test")
        list_chain_res = list(GA_chains.values())
        list_chain_keys = list(GA_chains.keys())
        list_fit_vals = np.array(fit_fun.f1_time_param_penalization(list_chain_res, query_params['a']))*-1
        sorted = np.argsort(list_fit_vals)
        for id in sorted:
            speedup_chain_id = list_chain_keys[id]
            speedup_chain_result = GA_chains[speedup_chain_id]
            if speedup_chain_result.test['system'].accuracy >= acc:
                break

        # Compute increase in params
        speedup.append(time/speedup_chain_result.test['system'].time)
        param_incrase.append(speedup_chain_result.test['system'].params/params)

    if len(GA_res_loc) > 0:
        print("Genetic Algorithm:")
        print("\tSize: %d" % len(GA_res_loc))
        print("\tAvg speedup:", stats.mean(speedup))
        print("\tAvg param incrase:", stats.mean(param_incrase))
    else:
        print("No data available")
