"""
EARN following GP genetic operations format
"""

from Examples.compute.bagging_boosting_of_chains_GA.main import *
del globals()["mutation_operation"]
del globals()["generate_offspring"]
import Examples.compute.bagging_boosting_of_chains_GA.main as main
import numpy as np


def mutation_operation(P):

    p = P[random.randint(0, len(P) - 1)]
    new_p = p.copy()
    new_p_components = np.array(list(new_p.get_message().classifier) + list(new_p.get_message().merger))  # Nodes of the ensemble DAG
    components_mutate = np.where(np.random.random(len(new_p_components)) < 1/len(new_p_components))[0]

    for component in new_p_components[components_mutate]:

        if component.DESCRIPTOR.name == "Merger":
            number_merged = len(component.merged_ids)
            operation = random.randint(0, 1) if number_merged < 3 else 1  # Limit the merged chains

            # Merger mutation: Add one more classifier
            if operation == 0:
                c_file_new = utils.pick_random_classifier(main.args)
                c_id_new = get_classifier_name(c_file_new)
                c_id_new = str(number_merged) + "_" + c_id_new
                om.add_classifier_to_merger(new_p, component.id, c_id_new, c_file_new)

            # Merger mutation: Change merging protocol
            # TODO: Change merging protocol

        elif component.DESCRIPTOR.name == "Classifier":
            operation = random.randint(0, 2) if component.component_id == "" else random.randint(1, 2)

            # Classifier mutation: Extend a chain
            if operation == 0:
                c_file_new = utils.pick_random_classifier(main.args)
                c_id_new = (component.id[0] + '_' if '0' < component.id[0] <= '9' else '') + get_classifier_name(c_file_new)
                om.extend_merged_chain(new_p, component.id, c_id_new, th=0.5, c_file_new=c_file_new)

            # Classifier mutation: Replace a classifier
            if operation == 1:
                c_file_new = utils.pick_random_classifier(main.args)
                c_id_new = (component.id[0] + '_' if '0' < component.id[0] <= '9' else '') + get_classifier_name(c_file_new)
                om.replace_classifier_merger(new_p, component.id, c_id_new, c_file=c_file_new)

            # Classifier mutation: Extend chain
            if operation == 2:
                new_p = p.copy()
                sign = 2*(random.random() > 0.5) - 1
                om.update_threshold(new_p, component.id, sign*main.args.step_th)

    return [new_p]


def generate_offspring(P, fit_vals, o=None):

    if o is None:
        o = main.args.offspring

    # Dictionary to avoid repeated offspring
    offspring_dict = {}
    while len(offspring_dict) < o:
        r = random.random()
        if main.args.pm > r:
            for offspring in mutation_operation(P):
                offspring_dict[offspring.get_sysid()] = offspring
        if main.args.pc > r:
            for offspring in crossover_operation_v2(P, fit_vals):
                offspring_dict[offspring.get_sysid()] = offspring

    return [offspring for key, offspring in offspring_dict.items()]


def argument_parse(argv):
    parser = argparse.ArgumentParser(usage="main.py [options]",
                                     description="Genetic algorithm for finding the best chain under environment constraints.")
    parser.add_argument("--dataset", default="sota_models_cifar10-32-dev_validation", help="Datasets to evaluate d1 d2 d3, default=*")
    # Search-space params
    parser.add_argument("--population", default=1000, type=int, help="Population at each generation")
    parser.add_argument("--offspring", default=50, type=int, help="Children generated at each generation")
    parser.add_argument("--iterations", default=50, type=int, help="Number of iterations before finishing algorithm")
    parser.add_argument("--step_th", default=0.1, type=float, help="Quant modification in threshold")
    parser.add_argument("--pm", default=0.9, type=float, help="Probability of mutation")
    parser.add_argument("--pc", default=0.1, type=float, help="Probability of crossing/breeding")
    parser.add_argument("--selection", default="nfit", type=str, help="Most fit selection (mfit) or roulette (roulette)")
    parser.add_argument("--a", nargs='+', default=[5/7, 1/7, 1/7], type=float, help="Fitting function's weight")
    parser.add_argument("--k", default=10, type=int, help="Tournament size")
    # Execution parameters
    parser.add_argument("--plot", default=0, type=int, help="Plot the ensembles generated every generation")
    parser.add_argument("--cores", default=0, type=int, help="Parallel evaluation of the ensembles")
    parser.add_argument("--device", default="none", type=str, help="Device where to execute the ensembles (cpu, gpu or none)")
    parser.add_argument("--comment", default="", type=str, help="")
    parser.add_argument("--experiment", default="main_GP.py")

    return parser.parse_args(argv)


if __name__ == "__main__":
    main.args = argument_parse(sys.argv[1:])

    # Create a temporal folder for the triggers models to be stored during the execution
    os.environ['TMP'] = 'Definitions/Classifiers/tmp/'+main.args.dataset[12:-15]
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

    fit = fit_fun.f2_time_param_penalization(R, main.args.a, limits)

    # Start the loop over generations
    iteration = 0
    while iteration < main.args.iterations:

        start = time.time()

        # Generate offspring and evaluate
        P_offspring = generate_offspring(P, fit)
        R_offspring = evaluate_population(P_offspring)
        fit_offspring = fit_fun.f2_time_param_penalization(R_offspring, main.args.a, limits)

        # Selection
        fit_generation = fit + fit_offspring
        P_generation = P + P_offspring
        R_generation = R + R_offspring
        if main.args.selection == "nfit":
            selected = selection.most_fit_selection(fit_generation, main.args.population)
        else:
            selected = selection.roulette_selection(fit_generation, main.args.population)

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
        print("Generation %d" % iteration)
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


