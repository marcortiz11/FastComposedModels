import Examples.paretto_front as front
import Examples.metadata_manager_results as results_manager
import Source.io_util as io
import statistics as stats
import os

if __name__ == "__main__":

    dataset = "sota_models_stl10-32-dev_validation"

    # 0) Single DNNs
    models = io.read_pickle(os.path.join(os.environ['FCM'], 'SmallSamples', 'models_evaluation', dataset, 'models.pkl'))
    models_front = front.get_front_time_accuracy(models, phase="test")
    sorted_models_front = front.sort_results_by_time(models_front, phase="test")

    # 1) G.A. Chain ensembles
    GA_results_metadata_file = os.path.join(os.environ['FCM'],
                                            'Examples',
                                            'Compute',
                                            'chain_genetic_algorithm',
                                            'results',
                                            'metadata.json')

    GA_ids = results_manager.get_ids_by_fieldval(GA_results_metadata_file, 'dataset', dataset)[-1:]

    speedup = []
    for id in GA_ids:
        local_path = results_manager.get_results_by_id(GA_results_metadata_file, id)
        global_path = os.path.join(os.environ['FCM'], 'Examples', 'Compute', 'chain_genetic_algorithm', local_path)
        GA_chains = io.read_pickle(global_path)
        front_GA_chain = front.get_front_time_accuracy(GA_chains, phase="test")
        sorted_front_GA_chain = front.sort_results_by_time(front_GA_chain, phase="test")
        speedup.append(front.get_speedup_front(sorted_models_front, sorted_front_GA_chain, e=1e6, phase="test"))

    if len(speedup) > 1:
        print("Genetic Algorithm:")
        print("\tAvg speedup:", stats.mean(speedup))
        print("\tMean speedup:", stats.median(speedup))
        print("\tStd speedup:", stats.stdev(speedup))

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
    front_BF_chains = front.get_front_time_accuracy(BF_chains, phase="test")
    sorted_front_BF_chains = front.sort_results_by_time(front_BF_chains, phase="test")
    speedup = front.get_speedup_front(sorted_models_front, sorted_front_BF_chains, phase="test", e=1e4)

    print("Brute-Force Speedup: %f" % speedup)
