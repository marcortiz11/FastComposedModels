import source.io_util as io
import source.genetic_algorithm.fitting_functions as fit_fun
import source.genetic_algorithm.selection as selection
import examples.compute.chain_genetic_algorithm.main as main
import source.system_evaluator as sys_eval
import argparse, sys, os, random, json, time, shutil

global args


def argument_parse(argv):
    parser = argparse.ArgumentParser(usage="main.py [options]",
                                     description="Genetic algorithm for finding the best chain under environment constraints.")
    parser.add_argument("--dataset", default="sota_models_cifar10-40-dev_validation", help="Datasets to evaluate d1 d2 d3, default=*")
    parser.add_argument("--population", default=100, type=int, help="Children generated at each generation")
    parser.add_argument("--offspring", default=100, type=int, help="Children generated at each generation")
    parser.add_argument("--pm", default=0.8, type=float, help="Probability of mutation")
    parser.add_argument("--pc", default=0.2, type=float, help="Probability of crossing/breeding")
    parser.add_argument("--step_th", default=0.05, type=float, help="Quant modification in threshold")
    parser.add_argument("--parallel", type=int, help="Parallel evaluation of the ensembles")
    parser.add_argument("--a", default=5, type=float)
    # Multi-Node execution parameters
    parser.add_argument("--node", default="n0", type=str)
    parser.add_argument("--selection", default=0, type=int)
    parser.add_argument("--experiment_id", default=0, type=str)

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
    experiment_id = str(main.args.experiment_id)  # Force all nodes to save to the same folder
    local_experiment_path = os.path.join('examples/compute/chain_genetic_algorithm_multinode/results', main.args.dataset,
                                   experiment_id)

    multinode_metainfo_file = os.path.join(os.environ['FCM'], local_experiment_path, 'multinode_metainfo.json')

    # Look if there is the multinode meta-information and results file
    if not os.path.exists(multinode_metainfo_file):
        os.makedirs(os.path.join(os.environ['FCM'], local_experiment_path))
        with open(multinode_metainfo_file, 'w') as handle:
            json.dump([{}], handle)

    initial_population = "./initial_population_0.pkl"
    if os.path.exists(os.path.join(os.environ['FCM'], local_experiment_path, initial_population)):
        files = [f for f in os.listdir(os.path.join(os.environ['FCM'], local_experiment_path)) if 'initial_population_' in f]
        initial_population = os.path.join(os.environ['FCM'], local_experiment_path, files[-1])  # Load last generation
    else:
        shutil.copyfile(initial_population, os.path.join(os.environ['FCM'], local_experiment_path, initial_population))

    # SELECT INDIVIDUALS
    if main.args.selection == 1:
        start = time.time()

        P, fit = load_initial_population(initial_population)

        # 1- Load and join partial results
        with open(multinode_metainfo_file, 'r') as handle:
            partial_results = json.load(handle)
            assert len(partial_results) > 0, "ERROR: No partial results to join"
            for k, v in partial_results[-1].items():
                P += io.read_pickle(os.path.join(os.environ['FCM'], v['offspring_fit_loc']))['P']
                fit += io.read_pickle(os.path.join(os.environ['FCM'], v['offspring_fit_loc']))['fit']

        # 2- Perform selection based on fit value
        selected = selection.most_fit_selection(fit, main.args.population)
        P = [P[i] for i in selected]
        fit = [fit[i] for i in selected]
        execution_time = time.time() - start

        iteration = (len(partial_results)-1)

        # Compute the evaluation results
        R_new_population = {}
        for p in P:
            R_new_population[p.get_sysid()] = sys_eval.evaluate(p, p.get_start())
        new_population_loc = os.path.join(local_experiment_path, 'R_population_%d.pkl' % iteration)

        # 3- Save results, Update meta_data with time required to join and add new entry to the meta_data
        with open(multinode_metainfo_file, 'w') as handle:
            partial_results[-1]["selection"] = {"exec_time": execution_time,
                                                "R":new_population_loc}
            io.save_pickle(os.path.join(os.environ['FCM'], new_population_loc), R_new_population)
            partial_results.append({})
            json.dump(partial_results, handle, indent=4)

        population_file = os.path.join(os.environ['FCM'], local_experiment_path, 'initial_population_%d.pkl' % iteration)
        io.save_pickle(population_file, {'P': P, 'fit': fit})

    # OFFSPRING GENERATE & EVALUATE:
    else:
        start = time.time()
        # 1- Load Population
        P, fit = load_initial_population(initial_population)
        # 2- Generate offspring (crossover+mutation)
        P_offspring = main.generate_offspring(P, fit)
        # 3- Evaluate offspring (Fitness)
        R_offspring = main.evaluate_population(P_offspring)
        fit_offspring = fit_fun.f1_time_penalization_preevaluated(R_offspring, a=main.args.a)
        execution_time = time.time() - start

        # Load the partial results and update it
        with open(multinode_metainfo_file) as handle:
            partial_results = json.load(handle)

        # Save offspring created by current node
        offspring = {'P': P_offspring, 'fit': fit_offspring}
        local_offspring_path = os.path.join(local_experiment_path, 'node-%s_iteration-%d.pkl' % (main.args.node, len(partial_results)-1))
        io.save_pickle(os.path.join(os.environ['FCM'], local_offspring_path), offspring)

        partial_results[-1][main.args.node] = {"exec_time": execution_time,
                                                "offspring_fit_loc": local_offspring_path}

        # Write to the meta-information json file
        with open(multinode_metainfo_file, 'w') as handle:
            json.dump(partial_results, handle, indent=4)





