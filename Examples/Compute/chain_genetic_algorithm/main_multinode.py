import Source.io_util as io
import Source.genetic_algorithm.fitting_functions as fit_fun
import Examples.Compute.chain_genetic_algorithm.main as main
import argparse, sys, os, random, json, time

global args


def argument_parse(argv):
    parser = argparse.ArgumentParser(usage="main.py [options]",
                                     description="Genetic algorithm for finding the best chain under environment constraints.")
    parser.add_argument("--dataset", default="front45_models_validation", help="Datasets to evaluate d1 d2 d3, default=*")
    parser.add_argument("--population", default=100, type=int, help="Population at each generation")
    parser.add_argument("--offspring", default=100, type=int, help="Children generated at each generation")
    parser.add_argument("--iterations", default=20, type=int, help="Number of iterations before finishing algorithm")
    parser.add_argument("--pm", default=0.8, type=float, help="Probability of mutation")
    parser.add_argument("--pc", default=0.2, type=float, help="Probability of crossing/breeding")
    parser.add_argument("--step_th", default=0.1, type=float, help="Quant modification in threshold")
    parser.add_argument("--parallel", type=int, help="Parallel evaluation of the ensembles")
    parser.add_argument("--a", default=100, type=float)
    # Multi-Node execution parameters
    parser.add_argument("--population_fitness", default="population_fitness.pkl", type=str)
    parser.add_argument("--meta_file", default="./results/metadata_multinode.json", type=str)
    parser.add_argument("--iteration", default=0, type=int)
    parser.add_argument("--node", default="n0", type=str)
    parser.add_argument("--selection", default=0, type=int)

    return parser.parse_args(argv)


def load_initial_population_and_fitness(file):
    assert 'P' in io.read_pickle(file) and 'fit' in io.read_pickle(file)
    P = io.read_pickle(file)['P']
    fit = io.read_pickle(file)['fit']
    return P, fit


if __name__ == "__main__":

    global args
    args = argument_parse(sys.argv[1:])

    start = time.time()
    random.seed()
    best_fit = []

    # 1- Load Population
    P, fit = load_initial_population_and_fitness(args.population_fitness)
    # 2- Generate offspring (crossover+mutation)
    P_offspring = main.generate_offspring(P, fit)
    # 3- Evaluate offspring (Fitness)
    R_offspring = main.evaluate_population(P_offspring)
    fit_offspring = fit_fun.f1_time_penalization_preevaluated(R_offspring, a=args.a, b=args.b)

    execution_time = time.time() - start

    """
    # Selection
    fit_generation = fit + fit_offspring
    P_generation = P + P_offspring
    selected = selection.roulette_selection(fit_generation, args.population)
    P = [P_generation[i] for i in selected]
    fit = [fit_generation[i] for i in selected]
    """

    # 4- Save the results
    import Examples.metadata_manager_results as manager_results

    meta_data_file = os.path.join(os.environ['FCM'],
                                  'Examples',
                                  'Compute',
                                  'chain_genetic_algorithm',
                                  'results',
                                  'metadata.json')

    id = str(random.randint(0, 1e8))
    results_loc = os.path.join('./results', args.dataset, id)
    comments = "PARTIAL RESULTS MULTI-NODE G.A. EXECUTION"
    meta_data_result = manager_results.metadata_template(id, args.dataset, results_loc, comments)
    params = args.__dict__
    manager_results.save_results(meta_data_file, meta_data_result, params, {'P': P_offspring, 'fit': fit_offspring})

    with open(args.meta_file, 'r') as handle:
        partial_results = json.load(handle)
        if len(partial_results) == 0:
            partial_results = [{}]

    partial_results[args.iteration][args.node]["exec_time"] = execution_time
    partial_results[args.iteration][args.node]["offspring_fit_loc"] = results_loc

    with open(args.meta_file, 'w') as handle:
        json.dump(partial_results, handle)





