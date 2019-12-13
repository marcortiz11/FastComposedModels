import Source.system_builder as sb
import Source.make_util as make
import Source.io_util as io
import Source.system_evaluator as ev
import Source.genetic_algorithm.selection as selection
import Source.genetic_algorithm.fitting_functions as fit_fun
import Source.genetic_algorithm.operations_mutation as om
import Source.genetic_algorithm.operations_breed as ob
import Examples.chain_genetic_algorithm.utils as utils
import argparse
import sys
import random
import os
import json

global args


def argument_parse(argv):
    parser = argparse.ArgumentParser(usage="main.py [options]",
                                     description="Genetic algorithm for finding the best chain under environment constraints.")
    parser.add_argument("--datasets", nargs="*", default="front45_models_validation", help="Datasets to evaluate d1 d2 d3, default=*")
    parser.add_argument("--population", default=100, type=int, help="Population at each generation")
    parser.add_argument("--offspring", default=100, type=int, help="Children generated at each generation")
    parser.add_argument("--iterations", default=20, type=int, help="Number of iterations before finishing algorithm")
    parser.add_argument("--pm", default=1, type=float, help="Probability of mutation")
    parser.add_argument("--pc", default=0.2, type=float, help="Probability of crossing/breeding")
    parser.add_argument("--time_constraint", default=10, type=float, help="Time constraint of the ensemble")
    parser.add_argument("--param_constraint", default=100000, type=float, help="Param constraint of the ensemble")
    parser.add_argument("--acc_constraint", default=0.8, type=float, help="Accuracy constraint")
    parser.add_argument("--step_th", default=0.1, type=float, help="Quant modification in threshold")
    parser.add_argument("--print", type=str, help="Folder in which to save the results of the execution")
    parser.add_argument("--a", default=100, type=float)
    parser.add_argument("--b", default=1, type=float)

    return parser.parse_args(argv)


def __get_classifier_name(c_file):
    return io.read_pickle(c_file)['name']


def __get_val_accuracy(c_file):
    s = sb.SystemBuilder()
    c = make.make_classifier("asdf",c_file)
    s.add_classifier(c)
    s.set_start("asdf")
    return ev.evaluate(s,s.get_start(),phases=["val"]).val['system'].accuracy


def generate_initial_population():
    classifier_path = os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers', args.datasets)
    P = []
    classifier_files = [os.path.join(classifier_path, f) for f in os.listdir(classifier_path) if ".pkl" in f]
    for c_file in classifier_files:
        sys_id = str(random.random())
        sys = sb.SystemBuilder(verbose=False, id=sys_id)
        c_id = __get_classifier_name(c_file)
        classifier = make.make_classifier(c_id, c_file)
        sys.add_classifier(classifier)
        sys.set_start(c_id)
        P.append(sys)
    return P


def mutation_operation(P, fit_vals):

    offspring = []
    p = P[selection.spin_roulette(fit_vals)]

    # Extend chain
    r = random.random()
    if args.pm > 0 and len(p.get_message().classifier) < 4:
        new_p = p.copy()
        new_p.set_sysid(str(random.random()))
        c_file = utils.pick_random_classifier(args)
        c_id = __get_classifier_name(c_file)
        om.extend_chain_pt(new_p, c_id, utils.pick_random_threshold(args), c_file=c_file)
        offspring.append(new_p)

    # Replace classifier
    r = random.random()
    if args.pm > 0:
        new_p = p.copy()
        new_p.set_sysid(str(random.random()))
        c_file_new = utils.pick_random_classifier(args)
        c_id_new = __get_classifier_name(c_file_new)
        om.replace_classifier(new_p, utils.pick_random_classifier(args, new_p), c_id_new, c_file=c_file_new)
        offspring.append(new_p)

    # Change threshold
    r = random.random()
    if args.pm > 0:
        new_p = p.copy()
        new_p.set_sysid(str(random.random()))
        sign = 2*(random.random() > 0.5) - 1
        om.update_threshold(new_p, utils.pick_random_classifier(args, new_p), sign*args.step_th,)
        offspring.append(new_p)

    return offspring


def crossover_operation(P, fit_vals):

    ai = selection.spin_roulette(fit_vals)
    bi = selection.spin_roulette(fit_vals)

    triggersA = P[ai].get_message().trigger
    triggersB = P[bi].get_message().trigger

    offspring = []

    if len(triggersA) > 0 and len(triggersB) > 0:
        pointA = triggersA[random.randint(0, len(triggersA)-1)].id
        pointB = triggersB[random.randint(0, len(triggersB)-1)].id
        offspring = ob.singlepoint_crossover(P[ai], P[bi], pointA, pointB)

    return offspring


def generate_offspring(P, fit_vals):
    offspring = []
    while len(offspring) < args.offspring:
        r = random.random()
        if args.pm > r:
            offspring += mutation_operation(P, fit_vals)
        if args.pc > r:
            offspring += crossover_operation(P, fit_vals)
    return offspring


def select_more_fit(P, fit, n):
    """
    Select and return the most fit individuals.
    :param P: Population
    :param fit: Fit value for each individual in P
    :param n: Number of allowed individuals in the next epoch
    :return: The n most fit individuals
    """
    selected = selection.roulette_selection(fit, n)
    new_P = [P[i] for i in selected]
    new_fit = [fit[i] for i in selected]
    return new_P, new_fit


if __name__ == "__main__":

    global args
    args = argument_parse(sys.argv[1:])
    random.seed()  # Initialize the random generator

    best_fit = []
    R = {}
    R_old = {}
    R_models = io.read_pickle(os.path.join(os.environ['FCM'], 'Examples', 'pushing_paretto_chain', 'results', 'front45_models_validation', 'models.pkl'))

    # Initial population
    P = generate_initial_population()
    fit = fit_fun.f1_time_penalization(P, time_constraint=args.time_constraint)

    # Start the loop over generations
    iteration = 0
    improving = True
    while iteration < args.iterations and improving:

        # Generate offspring (crossover+mutation)
        P_offspring = generate_offspring(P, fit)

        # Evaluate offspring individuals
        fit_offspring = fit_fun.f1_time_penalization(P_offspring, time_constraint=args.time_constraint, a=args.a, b=args.b)

        # Selection
        P, fit = select_more_fit(P+P_offspring, fit+fit_offspring, args.population)

        # Performance evaluation code
        best_fit.append(sum(fit)/len(fit))
        print("Iteration %d, average fit:%f" % (iteration, best_fit[-1]))

        # Plotting population
        R_old.update(R)
        R.clear()

        for p in P:
            R[p.get_sysid()] = ev.evaluate(p, p.get_start(), phases=["test"])
        utils.plot_population(R, R_old, R_models, iteration, args.time_constraint)

        iteration += 1

    # Save the results
    meta_data_file = os.path.join('./results', 'front45_models_validation', 'metadata.json')
    arguments_dict = args.__dict__
    meta_data_dict = {}

    if not os.path.exists(meta_data_file):
        with open(meta_data_file, 'w') as file:
            json.dump(meta_data_dict, file)

    with open(meta_data_file) as file:
        meta_data_dict = json.load(file)

    # Save results
    pid = str(os.getpid())
    res_dir = os.path.join('./results', 'front45_models_validation', pid)
    os.makedirs(res_dir)
    R_old.update(R)
    io.save_pickle(os.path.join(res_dir, 'R'), R_old)

    # Update meta_data file
    meta_data_dict[pid] = arguments_dict
    with open(meta_data_file, 'w') as file:
        json.dump(meta_data_dict, file, indent=4)


