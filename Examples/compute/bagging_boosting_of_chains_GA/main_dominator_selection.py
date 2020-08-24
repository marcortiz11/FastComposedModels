import Examples.compute.bagging_boosting_of_chains_GA.main as main
import Source.io_util as io
import Source.genetic_algorithm.selection as selection
import Source.genetic_algorithm.fitting_functions as fit_fun
import Source.genetic_algorithm.operations_breed as ob
import Examples.compute.chain_genetic_algorithm.utils as utils
import Source.system_evaluator as ev
import argparse, sys, random, os, time

def argument_parse(argv):
    parser = argparse.ArgumentParser(usage="main.py [options]",
                                     description="Genetic algorithm for finding the best chain under environment constraints.")
    parser.add_argument("--dataset", default="sota_models_cifar10-32-dev_validation", help="Datasets to evaluate d1 d2 d3, default=*")
    # Search-space params
    parser.add_argument("--iterations", default=200, type=int, help="Number of iterations before finishing algorithm")
    parser.add_argument("--step_th", default=0.1, type=float, help="Quant modification in threshold")
    parser.add_argument("--pm", default=1, type=float, help="Probability of mutation")
    parser.add_argument("--pc", default=0, type=float, help="Probability of crossing/breeding")
    parser.add_argument("--a", nargs='+', default=[1, 1, 0], type=float, help="Fitting function's weight")
    parser.add_argument("--offspring", default=50, type=int, help="Children generated at each generation")
    # Execution parameters
    parser.add_argument("--plot", default=0, type=int, help="Plot the ensembles generated every generation")
    parser.add_argument("--cores", default=0, type=int, help="Parallel evaluation of the ensembles")
    parser.add_argument("--device", default="none", type=str, help="Device where to execute the ensembles (cpu, gpu or none)")
    parser.add_argument("--comment", default="", type=str, help="")
    parser.add_argument("--experiment", default="bagging_boosting_of_chains_GA-dominator_selection")
    return parser.parse_args(argv)


def crossover_operation_v2(P):

    """
    Elitism parental selection and crossover operation.
    Two crossover operations:
        1) Merge two parents, if parents are chains.
        2) Exchange a part of the ensemble graph with single-point crossover, otherwise.
    :param P: Population of ensembles (Objects of SystemBuilder)
    :param fit_vals: Fitness values of the population (integer list)
    :return: The offspring
    """

    offspring = []
    a = P[random.randint(0, len(P) - 1)]
    b = P[random.randint(0, len(P) - 1)]

    # If two parents are chains merge them
    if len(a.get_message().merger) == 0 and len(b.get_message().merger) == 0:
        o = ob.merge_two_chains(a, b)
        offspring.append(o)

    # Single-Point Crossover of two Ensembles
    classifiersA = a.get_message().classifier
    classifiersB = b.get_message().classifier
    pointA = classifiersA[random.randint(0, len(classifiersA) - 1)].id
    pointB = classifiersB[random.randint(0, len(classifiersB) - 1)].id
    offspring += ob.singlepoint_crossover(a, b, pointA, pointB)

    for o in offspring:
        o.set_sysid(utils.generate_system_id(o))

    return offspring


def generate_offspring(P, fit_vals, o=None):

    if o is None:
        o = main.args.offspring

    # Dictionary to avoid repeated offspring
    offspring_dict = {}
    while len(offspring_dict) < o:
        r = random.random()
        if main.args.pm > r:
            for offspring in main.mutation_operation(P):
                offspring_dict[offspring.get_sysid()] = offspring
        if main.args.pc > r:
            for offspring in crossover_operation_v2(P):
                offspring_dict[offspring.get_sysid()] = offspring

    return [offspring for key, offspring in offspring_dict.items()]


def evaluate_process(P, pi, R, cores, phases):
    i = pi
    while i < len(P):
        R[i] = ev.evaluate(P[i], P[i].get_start(), phases=phases)
        i += cores


def evaluate_population(P, phases=['test', 'val']):

    R = [ev.Results]*len(P)

    if main.args.cores:

        assert main.args.device != 'cpu', "ERROR: Code does not evaluate the chain ensemble on PyTorch w\ CPU"

        from multiprocessing import Process, Manager
        processes = []
        R_manager = Manager().list(R)
        for i in range(main.args.cores):
            processes.append(Process(target=evaluate_process, args=(P, i, R_manager, main.args.cores, phases)))
            processes[i].start()
        for i in range(main.args.cores):
            processes[i].join()
        return list(R_manager)

    else:
        for i, p in enumerate(P):
            R[i] = ev.evaluate(p, p.get_start(), phases=phases)

            if main.args.device == 'cpu':
                from Source.iotnets.main_run_chain import chain_inference_time

                classifiers_chain = [utils.get_classifier_index(P[i], 0)] * 3
                ths = [0, 0]

                if len(P[i].get_message().classifier) > 1:
                    c_id = utils.get_classifier_index(P[i], 1)
                    t_id = P[i].get(classifiers_chain[0]).component_id
                    classifiers_chain[1] = c_id
                    ths[0] = float(t_id.split("_")[2])

                if len(P[i].get_message().classifier) > 2:
                    c_id = utils.get_classifier_index(P[i], 2)
                    t_id = P[i].get(classifiers_chain[1]).component_id
                    classifiers_chain[2] = c_id
                    ths[1] = float(t_id.split("_")[2])

                update = R[i]
                if 'val' in phases:
                    update.val['system'].time = chain_inference_time(main.args.dataset, classifiers_chain, ths, bs=128,
                                                                     phase='val')
                if 'test' in phases:
                    update.test['system'].time = chain_inference_time(main.args.dataset, classifiers_chain, ths, bs=128,
                                                                      phase='test')
                R[i] = update

    return R


if __name__ == "__main__":
    main.args = argument_parse(sys.argv[1:])
    os.environ['TMP'] = 'Definitions/Classifiers/tmp/'+main.args.dataset[12:-15]
    if not os.path.exists(os.path.join(os.environ['FCM'], os.environ['TMP'])):
        os.makedirs(os.path.join(os.environ['FCM'], os.environ['TMP']))
    random.seed()

    # Initial population
    P = main.generate_initial_population()
    R = main.evaluate_population(P)
    P_all = []
    individuals_fitness_per_generation = []

    # Evaluation Results Dictionary
    R_dict = {}
    R_dict_old = {}
    R_dict_models = dict(zip([p.get_sysid() for p in P], R))
    R_dict_old.update(R_dict_models)
    limits = fit_fun.make_limits_dict()
    fit_fun.update_limit_dict(limits, R_dict_models, phase="val")

    fit = fit_fun.f1_3objective_acc_time_param(R, main.args.a, limits)

    # Start the loop over generations
    iteration = 0
    p_update = main.args.pm/main.args.iterations
    while iteration < main.args.iterations:

        # Dynamic decreasing high mutation ratio (DHM)
        main.args.pm -= p_update
        main.args.pc += p_update

        start = time.time()

        # Generate offspring and evaluate
        P_offspring = generate_offspring(P, fit)
        R_offspring = evaluate_population(P_offspring)
        fit_offspring = fit_fun.f1_3objective_acc_time_param(R_offspring, main.args.a, limits)

        # Selection
        fit_generation = fit + fit_offspring
        P_generation = P + P_offspring
        R_generation = R + R_offspring
        selected = selection.dominator_selection(fit_generation)

        # Population Generation i+1
        P = [P_generation[i] for i in selected]
        fit = [fit_generation[i] for i in selected]
        R = [R_generation[i] for i in selected]

        # Plotting population
        R_dict_old.update(R_dict)
        R_dict.clear()
        for i, p in enumerate(P):
            R_dict[p.get_sysid()] = R[i]

        if main.args.plot:
            utils.plot_population(R_dict, R_dict_models, iteration)

        # Save which individuals alive every iteration
        ids = [p.get_sysid() for p in P]
        individuals_fitness_per_generation += [(ids, fit)]

        # Info about current generation
        print("Iteration %d" % iteration)
        print("TIME: Seconds per generation: %f " % (time.time()-start))

        iteration += 1

    # Save the results
    import Examples.metadata_manager_results as manager_results
    meta_data_file = os.path.join(os.environ['FCM'],
                                  'Examples',
                                  'compute',
                                  'bagging_boosting_of_chains_GA',
                                  'results',
                                  'metadata.json')

    id = str(random.randint(0, 1e16))
    results_loc = os.path.join('Examples/compute/bagging_boosting_of_chains_GA/results', main.args.dataset, id)
    comments = main.args.comment
    meta_data_result = manager_results.metadata_template(id, main.args.dataset, results_loc, comments)

    # Save the ensemble evaluation results
    R_dict_old.update(R_dict)
    params = main.args.__dict__
    manager_results.save_results(meta_data_file, meta_data_result, params, R_dict_old)
    io.save_pickle(os.path.join(os.environ['FCM'], results_loc, 'individuals_fitness_per_generation.pkl'),
                   individuals_fitness_per_generation)


