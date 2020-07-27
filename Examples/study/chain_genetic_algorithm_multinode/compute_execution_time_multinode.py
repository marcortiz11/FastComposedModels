import Source.io_util as io
import json

def get_n_maximum(iteration_partial_results, n_nodes):
    time = [r['exec_time'] for n, r in iteration_partial_results.items() if n != 'selection']
    time.sort(reverse=True)
    return time[:n_nodes]


if __name__ == "__main__":

    import os

    multinode_meta_file = os.path.join(os.environ['FCM'], 'Examples', 'compute', 'genetic_algorithm_multinode',
                                       'results', 'sota_models_cifar10-40-dev_validation', 'cifar10_8nodes_800population_400offspring_0',
                                       'multinode_metainfo.json')
    execution_time = 0
    n_nodes = 8
    R_all = {}

    with open(multinode_meta_file, 'r') as handle:
        results = json.load(handle)
        for i, iteration in enumerate(results):
            max_exec_offspring_time = sum(get_n_maximum(iteration, 1))
            selection_time = iteration['selection']['exec_time']
            execution_time += max_exec_offspring_time + selection_time

            # Gather the results
            R_iter = io.read_pickle(os.path.join(os.environ['FCM'], iteration['selection']['R']))
            R_all.update(R_iter)

            print("Iteration %d: Offspring+evaluate %f, Selection %f" % (i, max_exec_offspring_time, selection_time))
            
    print(R_all.keys())
    print("Execution time: %f" % execution_time)

    # Plot solutions
    import Examples.study.plot as myplt
    myplt.plot_accuracy_time_old(R_all)
    myplt.show()