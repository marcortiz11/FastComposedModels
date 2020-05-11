import source.system_builder_serializable as sb
import source.make_util as make
import source.io_util as io
import source.system_evaluator as ev
import source.genetic_algorithm.selection as selection
import source.genetic_algorithm.fitting_functions as fit_fun
import source.genetic_algorithm.operations_mutation as om
import source.genetic_algorithm.operations_breed as ob
import examples.compute.chain_genetic_algorithm.utils as utils
import argparse, sys, random, os, time
from source.iotnets.main_run_chain import chain_inference_time

global args


def argument_parse(argv):
    parser = argparse.ArgumentParser(usage="main.py [options]",
                                     description="Genetic algorithm for finding the best chain under environment constraints.")
    parser.add_argument("--dataset", default="sota_models_cifar10-32-dev_validation", help="Datasets to evaluate d1 d2 d3, default=*")
    # Search space params
    parser.add_argument("--population", default=1000, type=int, help="Population at each generation")
    parser.add_argument("--offspring", default=500, type=int, help="Children generated at each generation")
    parser.add_argument("--iterations", default=40, type=int, help="Number of iterations before finishing algorithm")
    parser.add_argument("--step_th", default=0.1, type=float, help="Quant modification in threshold")
    parser.add_argument("--pm", default=0.8, type=float, help="Probability of mutation")
    parser.add_argument("--pc", default=0.2, type=float, help="Probability of crossing/breeding")
    parser.add_argument("--selection", default="nfit", type=str, help="most fit selection (mfit) or roulette (roulette)")
    parser.add_argument("--a", default=5, type=float)
    # Execution parameters
    parser.add_argument("--plot", type=int, help="Plot the ensembles generated every generation")
    parser.add_argument("--parallel", type=int, help="Parallel evaluation of the ensembles")
    parser.add_argument("--comment", default="", type=str, help="Meaningful comments about the run")

    return parser.parse_args(argv)


def __get_classifier_name(c_file):
    return io.read_pickle(c_file)['name']


def generate_initial_population():
    classifier_path = os.path.join(os.environ['FCM'], 'definitions', 'Classifiers', args.dataset)
    P = []
    classifier_files = [os.path.join(classifier_path, f) for f in os.listdir(classifier_path) if ".pkl" in f]
    for c_file in classifier_files:
        sys = sb.SystemBuilder(verbose=False)
        c_id = __get_classifier_name(c_file)
        classifier = make.make_classifier(c_id, c_file)
        sys.add_classifier(classifier)
        sys.set_start(c_id)
        sys.set_sysid(utils.generate_system_id_chain(sys))
        P.append(sys)
    return P


def mutation_operation(P, fit_vals):

    offspring = []
    p = P[selection.spin_roulette(fit_vals)]

    # Extend chain
    if args.pm > 0 and len(p.get_message().classifier) < 3:
        new_p = p.copy()
        c_file = utils.pick_random_classifier(args)
        c_id = __get_classifier_name(c_file)
        om.extend_chain_pt(new_p, c_id, utils.pick_random_threshold(args), c_file=c_file)
        new_p.set_sysid(utils.generate_system_id_chain(new_p))
        offspring.append(new_p)

    # Replace classifier
    if args.pm > 0:
        new_p = p.copy()
        c_file_new = utils.pick_random_classifier(args)
        c_id_new = __get_classifier_name(c_file_new)
        om.replace_classifier(new_p, utils.pick_random_classifier(args, new_p), c_id_new, c_file=c_file_new)
        new_p.set_sysid(utils.generate_system_id_chain(new_p))
        offspring.append(new_p)

    # Update threshold
    if args.pm > 0:
        new_p = p.copy()
        sign = 2*(random.random() > 0.5) - 1
        om.update_threshold(new_p, utils.pick_random_classifier(args, new_p), sign*args.step_th)
        new_p.set_sysid(utils.generate_system_id_chain(new_p))
        offspring.append(new_p)

    return offspring


def crossover_operation(P, fit_vals):

    ai = selection.spin_roulette(fit_vals)
    bi = selection.spin_roulette(fit_vals)

    classifiersA = P[ai].get_message().classifier
    classifiersB = P[bi].get_message().classifier

    offspring = []

    if len(classifiersA) > 1 and len(classifiersB) > 1:
        random_number = random.randint(0, min(len(classifiersA)-1, len(classifiersB)-1))
        pointA = utils.get_classifier_index(P[ai], random_number)
        pointB = utils.get_classifier_index(P[bi], random_number)

        offspring = ob.singlepoint_crossover(P[ai], P[bi], pointA, pointB)
        for o in offspring:
            o.set_sysid(utils.generate_system_id_chain(o))  # Create ids for the offspring individuals
            assert len(o.get_message().classifier) < 4, "ERROR: Offspring > 3 classifiers chain"

    return offspring


def generate_offspring(P, fit_vals, o=None):

    if o is None:
        o = args.offspring

    offspring = []
    while len(offspring) < o:
        r = random.random()
        if args.pm > r:
            offspring += mutation_operation(P, fit_vals)
        if args.pc > r:
            offspring += crossover_operation(P, fit_vals)
    return offspring


def evaluate_process(P, pi, R, cores):
    i = pi
    while i < len(P):
        R[i] = ev.evaluate(P[i], P[i].get_start(), phases=["test", "val"])

        # Run DNN on CPU with batches of 128 images
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

        # Validation inference time 1 batch 128
        R[i].val['system'].time = chain_inference_time(args.dataset, classifiers_chain, ths)
        print("Individual %d running time: %f" % (i, R[i].val['system'].time))

        i += cores


def evaluate_population(P):
    """
    :param P: Population
    :return:
    """
    R = [ev.Results]*len(P)
    if args.parallel:
        from multiprocessing import Process, Manager
        processes = []
        R_manager = Manager().list(R)
        for i in range(args.parallel):
            processes.append(Process(target=evaluate_process, args=(P, i, R_manager, args.parallel)))
            processes[i].start()
        for i in range(args.parallel):
            processes[i].join()
        return list(R_manager)
    else:
        for i, p in enumerate(P):
            R[i] = ev.evaluate(p, p.get_start(), phases=["test", "val"])
    return R


if __name__ == "__main__":

    global args
    args = argument_parse(sys.argv[1:])
    os.environ['TMP'] = 'definitions/Classifiers/tmp/'+args.dataset[12:-15]
    if not os.path.exists(os.path.join(os.environ['FCM'], os.environ['TMP'])):
        os.makedirs(os.path.join(os.environ['FCM'], os.environ['TMP']))
    random.seed()

    R_dict = {}
    R_dict_old = {}
    R_dict_models = io.read_pickle(
        os.path.join(os.environ['FCM'], 'small_examples', 'models_evaluation', args.dataset, 'models.pkl'))

    # Initial population
    P = generate_initial_population()
    R = evaluate_population(P)
    fit = fit_fun.f1_time_param_penalization(R, a=args.a)
    P_all = []

    # Start the loop over generations
    iteration = 0
    while iteration < args.iterations:
        start = time.time()

        # Generate offspring (crossover+mutation)
        P_offspring = generate_offspring(P, fit)
        # Evaluate offspring
        R_offspring = evaluate_population(P_offspring)
        # Evaluate offspring individuals
        fit_offspring = fit_fun.f1_time_param_penalization(R_offspring, a=args.a)
        # Selection
        fit_generation = fit + fit_offspring
        P_generation = P + P_offspring
        R_generation = R + R_offspring
        if args.selection == "nfit":
            selected = selection.most_fit_selection(fit_generation, args.population)
        else:
            selected = selection.roulette_selection(fit_generation, args.population)
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

        # Info about current generation
        print("Iteration %d" % iteration)
        print("TIME: Seconds per generation: %f " % (time.time()-start))

        iteration += 1

    # Save the results
    import examples.metadata_manager_results as manager_results
    meta_data_file = os.path.join(os.environ['FCM'],
                                  'examples',
                                  'compute',
                                  'chain_genetic_algorithm',
                                  'results',
                                  'metadata.json')

    id = str(random.randint(0, 1e16))
    results_loc = os.path.join('examples/compute/chain_genetic_algorithm/results', args.dataset, id)
    comments = args.comment
    meta_data_result = manager_results.metadata_template(id, args.dataset, results_loc, comments)

    # Save the ensemble evaluation results
    R_dict_old.update(R_dict)
    params = args.__dict__
    manager_results.save_results(meta_data_file, meta_data_result, params, R_dict_old)


