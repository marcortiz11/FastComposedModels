import Examples.study.paretto_front as front
import Examples.metadata_manager_results as results_manager
import Source.io_util as io
import statistics as stats
import os

if __name__ == "__main__":

    dataset = "sota_models_caltech256-32-dev_validation"

    # 0) Single DNNs
    models = io.read_pickle(os.path.join(os.environ['FCM'], 'small_examples', 'models_evaluation', dataset, 'models.pkl'))
    models_front = front.get_front_time_accuracy(models, phase="test")
    sorted_models_front = front.sort_results_by_time(models_front, phase="test")

    # 1) G.A. Chain ensembles
    GA_results_metadata_file = os.path.join(os.environ['FCM'],
                                            'Examples',
                                            'compute',
                                            'chain_genetic_algorithm',
                                            'results',
                                            'metadata.json')

    GA_ids = results_manager.get_ids_by_fieldval(GA_results_metadata_file, 'dataset', dataset)[-3:]

    speedup = []
    for id in GA_ids:
        global_path = results_manager.get_results_by_id(GA_results_metadata_file, id)
        #global_path = os.path.join(os.environ['FCM'], 'examples', 'Compute', 'chain_genetic_algorithm', local_path)
        GA_chains = io.read_pickle(global_path)
        front_GA_chain = front.get_front_time_accuracy(GA_chains, phase="test")
        sorted_front_GA_chain = front.sort_results_by_time(front_GA_chain, phase="test")
        speedup.append(front.get_speedup_front(sorted_models_front, sorted_front_GA_chain, e=1e6, phase="test"))

    if len(speedup) == 1:
        speedup.append(speedup[-1])

    print("Genetic Algorithm:")
    print("\tAvg speedup:", stats.mean(speedup))
    print("\tMean speedup:", stats.median(speedup))
    print("\tStd speedup:", stats.stdev(speedup))

    # 2) Brute-force chain ensembles
    BF_results_metadata_file = os.path.join(os.environ['FCM'],
                                            'Examples',
                                            'compute',
                                            'fully_connected_chain',
                                            'results',
                                            'metadata.json')
    BF_id = results_manager.get_ids_by_fieldval(BF_results_metadata_file, 'dataset', dataset)[-1]

    local_path = results_manager.get_results_by_id(BF_results_metadata_file, BF_id)
    global_path = os.path.join(os.environ['FCM'], 'Examples', 'compute', 'fully_connected_chain', local_path)
    BF_chains = io.read_pickle(global_path)
    front_BF_chains = front.get_front_time_accuracy(BF_chains, phase="test")
    sorted_front_BF_chains = front.sort_results_by_time(front_BF_chains, phase="test")
    speedup = front.get_speedup_front(sorted_models_front, sorted_front_BF_chains, phase="test", e=1e4)

    print("Brute-Force Speedup: %f" % speedup)

    # 3) Speed-up chain with GPU main memory size constrain

    speedup = []

    for id in GA_ids:
        global_path = results_manager.get_results_by_id(GA_results_metadata_file, id)
        GA_chain_results = io.read_pickle(global_path)
        # Delete chains that do not fit in a single GPU
        keys = list(GA_chain_results.keys())
        for key in keys:
            if GA_chain_results[key].test['system'].params*4 > 16*1e9:
                del GA_chain_results[key]
        front_GA_chain = front.get_front_time_accuracy(GA_chain_results, phase="test")
        sorted_front_GA_chain = front.sort_results_by_time(front_GA_chain, phase="test")
        speedup.append(front.get_speedup_front(sorted_models_front, sorted_front_GA_chain, e=1e6, phase="test"))

    if len(speedup) == 1:
        speedup.append(speedup[-1])

    print("Genetic Algorithm GPU Memory Constrain:")
    print("\tAvg speedup:", stats.mean(speedup))
    print("\tMean speedup:", stats.median(speedup))
    print("\tStd speedup:", stats.stdev(speedup))