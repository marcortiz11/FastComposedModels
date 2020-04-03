import Examples.paretto_front as front
import Examples.metadata_manager_results as results_manager
import Source.io_util as io
import statistics as stats
import os

if __name__ == "__main__":

    dataset = "sota_models_food101-32-dev_validation"

    # 0) Single DNNs
    models = io.read_pickle(os.path.join(os.environ['FCM'], 'SmallSamples', 'models_evaluation', dataset, 'models.pkl'))
    models_front = front.get_front_time_accuracy(models, phase="test")
    sorted_models_front = front.sort_results_by_time(models_front, phase="test")
    accurate_NN_result = models[sorted_models_front[-1][0]]
    acc = accurate_NN_result.test['system'].accuracy
    time = accurate_NN_result.test['system'].time
    params = accurate_NN_result.test['system'].params
    print(sorted_models_front[-1][0])

    # 1) G.A. Chain ensembles
    GA_results_metadata_file = os.path.join(os.environ['FCM'],
                                            'Examples',
                                            'Compute',
                                            'chain_genetic_algorithm',
                                            'results',
                                            'metadata.json')

    GA_ids = results_manager.get_ids_by_fieldval(GA_results_metadata_file, 'dataset', dataset)[-3:-1]
    GA_ids = ["4985570220466883"]

    speedup = []
    param_incrase = []

    for id in GA_ids:
        global_path = results_manager.get_results_by_id(GA_results_metadata_file, id)
        GA_chains = io.read_pickle(global_path)

        # Compute speedup
        front_GA = front.get_front_time_accuracy(GA_chains, phase="test")
        sorted_GA_chain = front.sort_results_by_time(GA_chains, phase="test")

        # Obtenir la cadena amb accuracy >= best NN && min number params
        min_params = 1e50
        speedup_chain_id = ""
        for id in GA_chains:
            if GA_chains[id].test['system'].accuracy >= acc and \
                GA_chains[id].test['system'].time < time and \
                GA_chains[id].test['system'].params < min_params:
                min_params = GA_chains[id].test['system'].params
                speedup_chain_id = id

        # Retrieving the chain with the highest fit function
        import Source.genetic_algorithm.fitting_functions as fit_fun
        import numpy as np
        list_chain_res = list(GA_chains.values())
        list_chain_keys = list(GA_chains.keys())
        sorted = np.argsort(-1*fit_fun.f1_time_param_penalization(list_chain_res, 5))
        for id in sorted:
            speedup_chain_id = list_chain_keys[id]
            speedup_chain_result = GA_chains[speedup_chain_id]
            if speedup_chain_result.test['system'].accuracy >= acc:
                break

        # Compute increase in params
        speedup_chain_result = GA_chains[speedup_chain_id]
        speedup.append(time/speedup_chain_result.test['system'].time)
        param_incrase.append(min_params)

    if len(speedup) == 1:
        speedup.append(speedup[-1])

    print(speedup_chain_id)
    print("Genetic Algorithm:")
    print("\tAvg speedup:", stats.mean(speedup))
    print("\tAvg param incrase:", stats.mean(param_incrase)/accurate_NN_result.test['system'].params)


    # 2) Brute-force chain ensembles
    BF_results_metadata_file = os.path.join(os.environ['FCM'],
                                            'Examples',
                                            'Compute',
                                            'fully_connected_chain',
                                            'results',
                                            'metadata.json')
    BF_id = results_manager.get_ids_by_fieldval(BF_results_metadata_file, 'dataset', dataset)[-1]

    local_path = results_manager.get_results_by_id(BF_results_metadata_file, BF_id)
    global_path = os.path.join(os.environ['FCM'], 'Examples', 'Compute', 'fully_connected_chain', local_path)
    BF_chains = io.read_pickle(global_path)

    min_params = 1e100
    speedup_chain_id = ""
    for id in BF_chains:
        if BF_chains[id].test['system'].accuracy >= acc and \
                BF_chains[id].test['system'].time < time and \
                BF_chains[id].test['system'].params < min_params:
            min_params = BF_chains[id].test['system'].params
            speedup_chain_id = id

    front_BF_chains = front.get_front_time_accuracy(BF_chains, phase="test")
    sorted_front_BF_chains = front.sort_results_by_time(front_BF_chains, phase="test")
    speedup = time/BF_chains[speedup_chain_id].test['system'].time
    increment = BF_chains[speedup_chain_id].test['system'].params/params

    print("Brute-Force Speedup: %f" % speedup)
    print("Brute-Force Increment Params: %f" % increment)