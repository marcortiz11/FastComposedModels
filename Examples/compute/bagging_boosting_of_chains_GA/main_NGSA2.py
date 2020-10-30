"""
EARN following GP genetic operations format under NGSA2 elitism selection
"""

from Examples.compute.bagging_boosting_of_chains_GA.main import *
from Source.genetic_algorithm.fitting_functions import f1_3objective_acc_time_param as f
del globals()["mutation_operation"]
del globals()["generate_offspring"]
del globals()["crossover_operation_v2"]
import Examples.compute.bagging_boosting_of_chains_GA.main as main
import numpy as np


def mutation_operation(P: list) -> list:

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


def crossover_operation_v2(P:list, rank_dist:np.ndarray) -> list:

    assert len(P) > 0 and rank_dist.shape[1] == 2

    # Tournment selection of parent ensembles
    rank = rank_dist[:, 0]
    dist = rank_dist[:, 1]
    K = min(rank.shape[0]//2, main.args.k)

    ai = bi = 0
    while ai == bi:
        ai = selection.tournament_selection(rank, K, dist)  # Select according to rank and crowding distance
        bi = selection.tournament_selection(rank, K, dist)
    a = P[ai]
    b = P[bi]

    offspring = []

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

    return offspring


def generate_offspring(P:list, rank_dist: np.ndarray, o=None) -> list:

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
            for offspring in crossover_operation_v2(P, rank_dist):
                offspring_dict[offspring.get_sysid()] = offspring

    return [offspring for key, offspring in offspring_dict.items()]


def argument_parse(argv):
    parser = argparse.ArgumentParser(usage="main.py [options]",
                                     description="Genetic algorithm for finding the best chain under environment constraints.")
    parser.add_argument("--dataset", default="sota_models_cifar10-32-dev_validation", help="Datasets to evaluate d1 d2 d3, default=*")
    # Search-space params
    parser.add_argument("--population", default=500, type=int, help="Population at each generation")
    parser.add_argument("--offspring", default=100, type=int, help="Children generated at each generation")
    parser.add_argument("--iterations", default=50, type=int, help="Number of iterations before finishing algorithm")
    parser.add_argument("--step_th", default=0.1, type=float, help="Quant modification in threshold")
    parser.add_argument("--pm", default=0.8, type=float, help="Probability of mutation")
    parser.add_argument("--pc", default=0.2, type=float, help="Probability of crossing/breeding")
    parser.add_argument("--k", default=10, type=int, help="Tournament size")
    parser.add_argument("--a", nargs='+', default=[1, 1, 1], type=float, help="Fitting function's weight")
    # Execution parameters
    parser.add_argument("--plot", default=0, type=int, help="Plot the ensembles generated every generation")
    parser.add_argument("--cores", default=0, type=int, help="Parallel evaluation of the ensembles")
    parser.add_argument("--device", default="none", type=str, help="Device where to execute the ensembles (cpu, gpu or none)")
    parser.add_argument("--comment", default="", type=str, help="")
    parser.add_argument("--experiment", default="main_NGSA2.py")

    return parser.parse_args(argv)


def get_rank_crowding_dist(R: list, limits: dict) -> np.ndarray:
    rank = fast_non_dominated_sort(R)
    crowd_dist = np.zeros(rank.shape)
    for ri in np.unique(rank):
        positions = np.where(rank == ri)[0]
        Ri = [R[i] for i in positions]
        normalized_objectives = f(Ri, main.args.a, limits)
        crowd_dist[positions] = compute_crowding_distance(normalized_objectives)
    return np.column_stack((rank, -crowd_dist))


def fast_non_dominated_sort(R: list) -> np.ndarray:

    dominates = lambda r1, r2: r1.val["system"].accuracy >= r2.val["system"].accuracy and\
                               r1.val["system"].time <= r2.val["system"].time and \
                               r1.val["system"].params <= r2.val["system"].params

    rank = -np.ones(len(R), dtype=np.int)  # Rank of the solution
    N = -np.ones(rank.shape)  # Dominant solutions
    S = []  # Dominated solutions
    F = set()  # Current front

    for ri, r in enumerate(R):
        Sr = []
        n = 0
        for qi, q in enumerate(R):
            if ri != qi:
                if dominates(r, q):
                    Sr.append(qi)
                elif dominates(q, r):
                    n += 1
        S.append(Sr)
        N[ri] = n

        if n == 0:
            rank[ri] = 1
            F.add(ri)

    i = 1
    while len(F) > 0:
        Q = set()
        for ri in F:
            for qi in S[ri]:
                N[qi] -= 1
                if N[qi] == 0:
                    rank[qi] = i+1
                    Q.add(qi)
        i += 1
        F = Q

    return rank


def compute_crowding_distance(obj: np.ndarray) -> np.ndarray:
    D = np.zeros(obj.shape[0])
    for j in range(obj.shape[1]):
        obj_id_sorted = np.argsort(obj[:, j])
        D[obj_id_sorted[0]] = D[obj_id_sorted[-1]] = float("inf")
        for i in range(1, obj.shape[0] - 1):
            D[obj_id_sorted[i]] += obj[obj_id_sorted[i+1], j] - obj[obj_id_sorted[i-1], j]
    return D


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
    individuals_fitness_per_generation = []

    # Evaluation Results Dictionary
    R_dict = {}
    R_dict_old = {}
    R_dict_models = dict(zip([p.get_sysid() for p in P], R))
    R_dict_old.update(R_dict_models)
    limits = fit_fun.make_limits_dict()
    fit_fun.update_limit_dict(limits, R_dict_models, phase="val")

    rank_dist = get_rank_crowding_dist(R, limits)  # Numpy array

    # Start the loop over generations
    iteration = 0
    while iteration < main.args.iterations:

        start = time.time()

        # Generate offspring and evaluate
        P_offspring = generate_offspring(P, rank_dist)
        R_offspring = evaluate_population(P_offspring)

        # Selection
        P_generation = P + P_offspring
        R_generation = R + R_offspring
        rank_dist_generation = get_rank_crowding_dist(R_generation, limits)

        selected = np.lexsort(rank_dist_generation.T[::-1])[:main.args.population]
        P = [P_generation[i] for i in selected]
        R = [R_generation[i] for i in selected]
        rank_dist = rank_dist_generation[selected]

        # Plotting population
        R_dict_old.update(R_dict)
        R_dict.clear()
        for i, p in enumerate(P):
            R_dict[p.get_sysid()] = R[i]
        if main.args.plot:
            utils.plot_population(R_dict, R_dict_models, iteration)

        # Save which individuals alive every iteration
        ids = [p.get_sysid() for p in P]
        individuals_fitness_per_generation += [(ids, rank_dist)]

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


