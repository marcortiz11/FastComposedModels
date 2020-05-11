import examples.paretto_front as front
import examples.metadata_manager_results as results_manager
import source.io_util as io
import statistics as stats
import os

if __name__ == "__main__":

    dataset = "sota_models_caltech256-32-dev_validation"
    GA_ids = ["8436913852312834", "5322753737898232", "9311417823719622"]

    # 0) Single DNNs
    models = io.read_pickle(os.path.join(os.environ['FCM'], 'small_examples', 'models_evaluation', dataset, 'models.pkl'))
    models_front = front.get_front_time_accuracy(models, phase="test")
    sorted_models_front = front.sort_results_by_time(models_front, phase="test")
    accurate_NN_result = models[sorted_models_front[-1][0]]
    print(sorted_models_front[-1][0])

    # 1) G.A. Chain ensembles
    GA_results_metadata_file = os.path.join(os.environ['FCM'],
                                            'examples',
                                            'compute',
                                            'chain_genetic_algorithm',
                                            'results',
                                            'metadata.json')

    #GA_ids = results_manager.get_ids_by_fieldval(GA_results_metadata_file, 'dataset', dataset)[-3:]

    speedup = []
    param_incrase = []

    for id in GA_ids:
        global_path = results_manager.get_results_by_id(GA_results_metadata_file, id)
        GA_chains = io.read_pickle(global_path)

        # Compute speedup
        front_GA_chain = front.get_front_time_accuracy(GA_chains, phase="test")
        sorted_front_GA_chain = front.sort_results_by_time(front_GA_chain, phase="test")
        speedup.append(front.get_speedup_front(sorted_models_front, sorted_front_GA_chain, e=1e9, phase="test"))


        # Compute increase in params
        speedup_chain_id = front.get_speedup_id(sorted_models_front, sorted_front_GA_chain, e=1e9, phase="test")
        speedup_chain_result = GA_chains[speedup_chain_id]
        param_incrase.append(speedup_chain_result.test['system'].params)

    if len(speedup) == 1:
        speedup.append(speedup[-1])

    print("Genetic Algorithm:")
    print("\tChain: %s" % speedup_chain_id)
    print("\tAvg speedup:", stats.mean(speedup))
    print("\tAvg param incrase:", stats.mean(param_incrase)/accurate_NN_result.test['system'].params)

    # 2) Brute-force chain ensembles
    BF_results_metadata_file = os.path.join(os.environ['FCM'],
                                            'examples',
                                            'compute',
                                            'fully_connected_chain',
                                            'results',
                                            'metadata.json')
    BF_id = results_manager.get_ids_by_fieldval(BF_results_metadata_file, 'dataset', dataset)[-1]

    local_path = results_manager.get_results_by_id(BF_results_metadata_file, BF_id)
    global_path = os.path.join(os.environ['FCM'], 'examples', 'compute', 'fully_connected_chain', local_path)
    BF_chains = io.read_pickle(global_path)
    front_BF_chains = front.get_front_time_accuracy(BF_chains, phase="test")
    sorted_front_BF_chains = front.sort_results_by_time(front_BF_chains, phase="test")
    speedup = front.get_speedup_front(sorted_models_front, sorted_front_BF_chains, phase="test", e=1e6)
    speedup_id = front.get_speedup_id(sorted_models_front, sorted_front_BF_chains, phase="test", e=1e6)

    print("Brute-Force chain: %s" % speedup_id)
    print("Brute-Force Speedup: %f" % speedup)
    print("Brute-Force Param Increase: %f" % (BF_chains[speedup_id].test['system'].params/accurate_NN_result.test['system'].params))
