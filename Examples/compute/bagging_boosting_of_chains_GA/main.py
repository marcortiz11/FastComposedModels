import Source.system_builder_serializable as sb
import Source.make_util as make
import Source.io_util as io
import Source.system_evaluator as ev
import Source.genetic_algorithm.selection as selection
import Source.genetic_algorithm.fitting_functions as fit_fun
import Source.genetic_algorithm.operations_mutation as om
import Source.genetic_algorithm.operations_breed as ob
import Examples.compute.chain_genetic_algorithm.utils as utils
import argparse, sys, random, os, time

global args


def argument_parse(argv):
    parser = argparse.ArgumentParser(usage="main.py [options]",
                                     description="Genetic algorithm for finding the best chain under environment constraints.")
    parser.add_argument("--dataset", default="sota_models_cifar10-32-dev_validation", help="Datasets to evaluate d1 d2 d3, default=*")
    # Search-space params
    parser.add_argument("--population", default=1000, type=int, help="Population at each generation")
    parser.add_argument("--offspring", default=500, type=int, help="Children generated at each generation")
    parser.add_argument("--iterations", default=40, type=int, help="Number of iterations before finishing algorithm")
    parser.add_argument("--step_th", default=0.1, type=float, help="Quant modification in threshold")
    parser.add_argument("--pm", default=0.9, type=float, help="Probability of mutation")
    parser.add_argument("--pc", default=0.1, type=float, help="Probability of crossing/breeding")
    parser.add_argument("--selection", default="nfit", type=str, help="Most fit selection (mfit) or roulette (roulette)")
    parser.add_argument("--a", nargs='+', default=[5/7, 1/7, 1/7], type=float, help="Fitting function's weight")
    parser.add_argument("--k", default=50, type=int, help="Tournament size")
    # Execution parameters
    parser.add_argument("--plot", default=0, type=int, help="Plot the ensembles generated every generation")
    parser.add_argument("--cores", default=0, type=int, help="Parallel evaluation of the ensembles")
    parser.add_argument("--device", default="none", type=str, help="Device where to execute the ensembles (cpu, gpu or none)")
    parser.add_argument("--comment", default="", type=str, help="")
    parser.add_argument("--experiment", default="bagging_boosting_of_chains_GA_1")

    return parser.parse_args(argv)


def __get_classifier_name(c_file):
    return io.read_pickle(c_file)['name']


def generate_initial_population():
    classifier_path = os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers', args.dataset)
    P = []
    classifier_files = [os.path.join(classifier_path, f) for f in os.listdir(classifier_path) if ".pkl" in f]
    for c_file in classifier_files:
        sys = sb.SystemBuilder(verbose=False)
        c_id = __get_classifier_name(c_file)
        classifier = make.make_classifier(c_id, c_file)
        sys.add_classifier(classifier)
        sys.set_start(c_id)
        sys.set_sysid(utils.generate_system_id(sys))
        P.append(sys)
    return P


def mutation_operation(P):

    offspring = []
    p = P[random.randint(0, len(P) - 1)]

    operation = random.randint(0, 3) if len(p.get_message().merger) == 1 else random.randint(0, 2)

    # Extend a chain
    if operation == 0:
        new_p = p.copy()

        # Pick a random classifier from the pool of solutions
        c_file_new = utils.pick_random_classifier(args)
        c_id_new = __get_classifier_name(c_file_new)

        # Find the tail of any chain to extend
        merger = None if 'Merger' not in new_p.get_start() else new_p.get_message().merger[0]
        n_chains = len(merger.merged_ids) if merger is not None else 1
        tail_chains = [c.id for c in new_p.get_message().classifier if c.component_id == ""]
        c_id_extend = tail_chains[random.randint(0, n_chains - 1)]
        c_id_new = (c_id_extend[0] + '_' if c_id_extend[0] > '0' and c_id_extend[0] <= '9' else '') + c_id_new

        # Perform the operation
        om.extend_merged_chain(new_p, c_id_extend, c_id_new, th=utils.pick_random_threshold(args), c_file_new=c_file_new)
        new_p.set_sysid(utils.generate_system_id(new_p))
        offspring.append(new_p)

    # Replace a classifier
    if operation == 1:
        new_p = p.copy()
        c_id_existing = utils.pick_random_classifier(args, new_p)
        c_file_new = utils.pick_random_classifier(args)
        c_id_new = (c_id_existing[0] + '_' if c_id_existing[0] > '0' and c_id_existing[0] <= '9' else '') +__get_classifier_name(c_file_new)
        om.replace_classifier_merger(new_p, c_id_existing, c_id_new, c_file=c_file_new)
        new_p.set_sysid(utils.generate_system_id(new_p))
        offspring.append(new_p)

    # Update threshold
    if operation == 2:
        new_p = p.copy()
        sign = 2*(random.random() > 0.5) - 1
        om.update_threshold(new_p, utils.pick_random_classifier(args, new_p), sign*args.step_th)
        new_p.set_sysid(utils.generate_system_id(new_p))
        offspring.append(new_p)

    # Add classifier to be merged (Assuming 1 merger)
    if operation == 3:
        merger = p.get_message().merger[0]
        merger_id = merger.id
        merger_number_chains = len(merger.merged_ids)

        if merger_number_chains < 3:
            new_p = p.copy()

            c_file_new = utils.pick_random_classifier(args)
            c_id_new = __get_classifier_name(c_file_new)
            c_id_new = str(merger_number_chains) + "_" + c_id_new

            om.add_classifier_to_merger(new_p, merger_id, c_id_new, c_file_new)
            new_p.set_sysid(utils.generate_system_id(new_p))
            offspring.append(new_p)

    return offspring


def crossover_operation(P, fit_vals):

    indices = [i for i, p in enumerate(P) if len(p.get_message().merger) == 0]
    P_chains = [P[i] for i in indices]
    Fit_chains = [fit_vals[i] for i in indices]

    offspring = []

    if len(indices) > 1:

        ai = bi = 0
        while ai == bi:
            ai = selection.tournament_selection(Fit_chains, min(len(Fit_chains)//2, args.k))
            bi = selection.tournament_selection(Fit_chains, min(len(Fit_chains)//2, args.k))

        # i_chains = random.sample(range(0, len(P_chains)), 2)

        a = P_chains[ai]
        b = P_chains[bi]

        # Crossover operation: Merge chains or single classifiers
        o = ob.merge_two_chains(a, b)
        o.set_sysid(utils.generate_system_id(o))
        offspring.append(o)

    return offspring


def crossover_operation_v2(P, fit_vals):

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

    ai = bi = 0
    while ai == bi:
        ai = selection.tournament_selection(fit_vals, min(len(fit_vals)//2, args.k))
        bi = selection.tournament_selection(fit_vals, min(len(fit_vals)//2, args.k))
    a = P[ai]
    b = P[bi]

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
        o = args.offspring

    # Dictionary to avoid repeated offspring
    offspring_dict = {}
    while len(offspring_dict) < o:
        r = random.random()
        if args.pm > r:
            for offspring in mutation_operation(P):
                offspring_dict[offspring.get_sysid()] = offspring
        if args.pc > r:
            for offspring in crossover_operation_v2(P, fit_vals):
                offspring_dict[offspring.get_sysid()] = offspring

    return [offspring for key, offspring in offspring_dict.items()]


def evaluate_process(P, pi, R, cores, phases):
    i = pi
    while i < len(P):
        R[i] = ev.evaluate(P[i], P[i].get_start(), phases=phases)
        i += cores


def evaluate_population(P, phases=['test', 'val']):

    R = [ev.Results]*len(P)

    if args.cores:

        assert args.device != 'cpu', "ERROR: Code does not evaluate the chain ensemble on PyTorch w\ CPU"

        from multiprocessing import Process, Manager
        processes = []
        R_manager = Manager().list(R)
        for i in range(args.cores):
            processes.append(Process(target=evaluate_process, args=(P, i, R_manager, args.cores, phases)))
            processes[i].start()
        for i in range(args.cores):
            processes[i].join()
        return list(R_manager)

    else:
        for i, p in enumerate(P):
            R[i] = ev.evaluate(p, p.get_start(), phases=phases)

            if args.device == 'cpu':
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
                    update.val['system'].time = chain_inference_time(args.dataset, classifiers_chain, ths, bs=128,
                                                                     phase='val')
                if 'test' in phases:
                    update.test['system'].time = chain_inference_time(args.dataset, classifiers_chain, ths, bs=128,
                                                                      phase='test')
                R[i] = update

    return R


if __name__ == "__main__":

    global args
    args = argument_parse(sys.argv[1:])
    os.environ['TMP'] = 'Definitions/Classifiers/tmp/'+args.dataset[12:-15]
    if not os.path.exists(os.path.join(os.environ['FCM'], os.environ['TMP'])):
        os.makedirs(os.path.join(os.environ['FCM'], os.environ['TMP']))
    random.seed()

    # Initial population
    P = generate_initial_population()
    R = evaluate_population(P)
    P_all = []
    individuals_fitness_per_generation = []

    # Evaluation Results Dictionary
    R_dict = {}
    R_dict_old = {}
    R_dict_models = dict(zip([p.get_sysid() for p in P], R))
    R_dict_old.update(R_dict_models)
    limits = fit_fun.make_limits_dict()
    fit_fun.update_limit_dict(limits, R_dict_models, phase="val")

    fit = fit_fun.f2_time_param_penalization(R, args.a, limits)

    # Start the loop over generations
    iteration = 0
    p_update = args.pm/args.iterations
    while iteration < args.iterations:

        # Dynamic decreasing high mutation ratio (DHM)
        args.pm -= p_update
        args.pc += p_update

        start = time.time()

        # Generate offspring and evaluate
        P_offspring = generate_offspring(P, fit)
        R_offspring = evaluate_population(P_offspring)
        fit_offspring = fit_fun.f2_time_param_penalization(R_offspring, args.a, limits)

        # Selection
        fit_generation = fit + fit_offspring
        P_generation = P + P_offspring
        R_generation = R + R_offspring
        if args.selection == "nfit":
            selected = selection.most_fit_selection(fit_generation, args.population)
        else:
            selected = selection.roulette_selection(fit_generation, args.population)

        # Population Generation i+1
        P = [P_generation[i] for i in selected]
        fit = [fit_generation[i] for i in selected]
        R = [R_generation[i] for i in selected]

        # Plotting population
        R_dict_old.update(R_dict)
        R_dict.clear()
        for i, p in enumerate(P):
            R_dict[p.get_sysid()] = R[i]
        if args.plot:
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
    results_loc = os.path.join('Examples/compute/bagging_boosting_of_chains_GA/results', args.dataset, id)
    comments = args.comment
    meta_data_result = manager_results.metadata_template(id, args.dataset, results_loc, comments)

    # Save the ensemble evaluation results
    R_dict_old.update(R_dict)
    params = args.__dict__
    manager_results.save_results(meta_data_file, meta_data_result, params, R_dict_old)
    io.save_pickle(os.path.join(os.environ['FCM'], results_loc, 'individuals_fitness_per_generation.pkl'),
                   individuals_fitness_per_generation)


