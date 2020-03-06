import Source.io_util as io
import Source.genetic_algorithm.fitting_functions as fit_fun
import Source.genetic_algorithm.selection as selection
import Examples.Compute.chain_genetic_algorithm.main as main
import argparse, sys, os, random, json, time

global args


def argument_parse(argv):
    parser = argparse.ArgumentParser(usage="main.py [options]",
                                     description="Genetic algorithm for finding the best chain under environment constraints.")
    parser.add_argument("--dataset", default="sota_models_cifar10-32-dev_validation", help="Datasets to evaluate d1 d2 d3, default=*")
    parser.add_argument("--population", default=100, type=int, help="Children generated at each generation")
    parser.add_argument("--offspring", default=100, type=int, help="Children generated at each generation")
    parser.add_argument("--pm", default=0.8, type=float, help="Probability of mutation")
    parser.add_argument("--pc", default=0.2, type=float, help="Probability of crossing/breeding")
    parser.add_argument("--step_th", default=0.1, type=float, help="Quant modification in threshold")
    parser.add_argument("--parallel", type=int, help="Parallel evaluation of the ensembles")
    parser.add_argument("--a", default=50, type=float)
    # Multi-Node execution parameters
    parser.add_argument("--initial_population", default="initial_population.pkl", type=str)
    parser.add_argument("--meta_file", default="./results/metadata_multinode.json", type=str)
    parser.add_argument("--node", default="n0", type=str)
    parser.add_argument("--selection", default=0, type=int)

    return parser.parse_args(argv)


def load_initial_population(file):
    assert 'P' in io.read_pickle(file) and 'fit' in io.read_pickle(file)
    P = io.read_pickle(file)['P']
    fit = io.read_pickle(file)['fit']
    return P, fit


if __name__ == "__main__":

    main.args = argument_parse(sys.argv[1:])
    random.seed()
    best_fit = []

    # SELECT POPULATION FROM PARTIAL RESULTS
    if main.args.selection == 1:

        P, fit = load_initial_population(main.args.initial_population)

        # 1- Load and join partial results
        with open(main.args.meta_file, 'r') as handle:
            partial_results = json.load(handle)
            assert len(partial_results) > 0, "ERROR: No partial results to join"
            for k, v in partial_results[-1].items():
                P += io.read_pickle(v['offspring_fit_loc']+"/ensembles.pkl")['P']
                fit += io.read_pickle(v['offspring_fit_loc']+"/ensembles.pkl")['fit']

        # 2- Perform selection based on fit value
        start = time.time()
        selected = selection.most_fit_selection(fit, main.args.population)
        P = [P[i] for i in selected]
        fit = [fit[i] for i in selected]
        execution_time = time.time() - start

        # 3- Save results, Update meta_data with time required to join and add new entry to the meta_data
        with open(main.args.meta_file, 'w') as handle:
            partial_results[-1]["selection"] = {"exec_time": execution_time}
            partial_results.append({})
            json.dump(partial_results, handle, indent=4)

        iteration = (len(partial_results)-1)
        io.save_pickle('./initial_population_%d.pkl' % iteration, {'P': P, 'fit': fit})

    # PARTIAL RESULT:
    else:
        start = time.time()
        # 1- Load Population
        P, fit = load_initial_population(main.args.initial_population)
        # 2- Generate offspring (crossover+mutation)
        P_offspring = main.generate_offspring(P, fit)
        # 3- Evaluate offspring (Fitness)
        R_offspring = main.evaluate_population(P_offspring)
        fit_offspring = fit_fun.f1_time_penalization_preevaluated(R_offspring, a=main.args.a)
        execution_time = time.time() - start

        # 4- Save the results
        import Examples.metadata_manager_results as manager_results

        meta_data_file = os.path.join(os.environ['FCM'],
                                      'Examples',
                                      'Compute',
                                      'chain_genetic_algorithm',
                                      'results',
                                      'metadata.json')

        id = str(random.randint(0, 1e16))
        results_loc = os.path.join('Examples/Compute/chain_genetic_algorithm/results', main.args.dataset, id)
        comments = "PARTIAL RESULTS MULTI-NODE G.A. EXECUTION"
        meta_data_result = manager_results.metadata_template(id, main.args.dataset, results_loc, comments)
        params = main.args.__dict__
        manager_results.save_results_and_ensembles(meta_data_file, meta_data_result, params, {}, {'P': P_offspring, 'fit': fit_offspring})

        # Update multi-node metadata file
        with open(main.args.meta_file, 'r') as handle:
            partial_results = json.load(handle)
            if len(partial_results) == 0:
                partial_results = [{}]

        partial_results[-1][main.args.node] = {"exec_time": execution_time,
                                                "offspring_fit_loc": results_loc}
        with open(main.args.meta_file, 'w') as handle:
            json.dump(partial_results, handle, indent=4)





