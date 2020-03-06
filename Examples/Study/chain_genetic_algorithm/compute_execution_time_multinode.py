import json


def get_n_maximum(iteration_partial_results, n_nodes):
    time = [r['exec_time'] for n, r in iteration_partial_results.items() if n != 'selection']
    time.sort(reverse=True)
    return time[:n_nodes]


if __name__ == "__main__":

    import os

    multinode_meta_file = os.path.join(os.environ['FCM'], 'Examples', 'Compute', 'chain_genetic_algorithm', 'results', 'metadata_multinode.json')
    execution_time = 0
    n_nodes = 128

    with open(multinode_meta_file, 'r') as handle:
        results = json.load(handle)
        for i, iteration in enumerate(results):
            max_exec_offspring_time = sum(get_n_maximum(iteration, int(128/n_nodes)))
            selection_time = iteration['selection']['exec_time']
            execution_time += max_exec_offspring_time + selection_time
            print("Iteration %d: Offspring+evaluate %f, Selection %f" % (i, max_exec_offspring_time, selection_time))

    print("Execution time: %f" % execution_time)