import Examples.paretto_front as front_tools
import Examples.fully_connected_chain.results_study as rs
import Source.system_builder as sb
import Source.make_util as make
import Source.io_util as io
import Source.system_evaluator as ev
import Source.genetic_algorithm.selection as selection
import Source.genetic_algorithm.fitting_functions as fit_fun
import Source.genetic_algorithm.operations_mutation as om
import argparse
import sys
import matplotlib.pyplot as plt
import random
import os

global args


def argument_parse(argv):
    parser = argparse.ArgumentParser(usage="main.py [options]",
                                     description="Genetic algorithm for finding the best chain under environment constraints.")
    parser.add_argument("--datasets", nargs="*", default="front45_models_validation", help="Datasets to evaluate d1 d2 d3, default=*")
    parser.add_argument("--population", default=100, type=int, help="Population at each generation")
    parser.add_argument("--iterations", default=20, type=int, help="Number of iterations before finishing algorithm")
    parser.add_argument("--pm", default=0.1, type=float, help="Probability of mutation")
    parser.add_argument("--pc", default=0.2, type=float, help="Probability of crossing/breeding")
    parser.add_argument("--time_constraint", default=10, type=float, help="Time constraint of the ensemble")
    parser.add_argument("--param_constraint", default=100000, type=float, help="Param constraint of the ensemble")
    parser.add_argument("--acc_constraint", default=0.8, type=float, help="Accuracy constraint")
    parser.add_argument("--step_th", default=0.1, type=float, help="Quant modification in threshold")
    return parser.parse_args(argv)


def __get_classifier_name(c_file):
    return io.read_pickle(c_file)['name']

def __get_val_accuracy(c_file):
    s = sb.SystemBuilder()
    c = make.make_classifier("asdf",c_file)
    s.add_classifier(c)
    s.set_start("asdf")
    return ev.evaluate(s,s.get_start(),phases=["val"]).val['system'].accuracy

# Generate initial population
def generate_initial_population():
    # 1- Load classifier files from the dataset
    # 2- Build a system with a single classifier for each one
    # 3- Return the population
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


def pick_random_threshold():
    return 0.5


def pick_random_classifier(sys=None, c_file=None):
    classifier_path = os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers', args.datasets)
    classifier_files = [os.path.join(classifier_path, f) for f in os.listdir(classifier_path) if ".pkl" in f]
    selected = classifier_files[random.randint(0, len(classifier_files)-1)]
    # Pick only classifiers with higher accuracy
    if sys is not None:
        classifier_files = sys.get_message().classifier
        selected = classifier_files[random.randint(0, len(classifier_files)-1)].id
    return selected


# Apply mutations to the current population
def mutation_operations(P):
    # Update counter for the children
    # c_id = c_file if possible
    P_mutation = []
    for p in P:
        # Extend chain
        r = random.random()
        if args.pm > r and len(p.get_message().classifier) < 4:
            new_p = p.copy()
            new_p.set_sysid(str(random.random()))
            c_file = pick_random_classifier()
            c_id = __get_classifier_name(c_file)
            om.extend_chain_pt(new_p, c_id, pick_random_threshold(), c_file=c_file)
            P_mutation.append(new_p)
        # Replace classifier
        r = random.random()
        if args.pm > r:
            new_p = p.copy()
            new_p.set_sysid(str(random.random()))
            c_file_new = pick_random_classifier()
            c_id_new = __get_classifier_name(c_file_new)
            om.replace_classifier(new_p, pick_random_classifier(new_p), c_id_new, c_file=c_file_new)
            P_mutation.append(new_p)
        # Change threshold
        r = random.random()
        if args.pm > 0:
            new_p = p.copy()
            new_p.set_sysid(str(random.random()))
            sign = 2*(random.random() > 0.5) - 1
            om.update_threshold(new_p, pick_random_classifier(new_p), sign*args.step_th,)
            P_mutation.append(new_p)
    return P+P_mutation


# Apply crossover operations
def breed_operations(P):
    return P


def select_more_fit(P, fit, n):
    """
    Select and return the most fit individuals.
    :param P: Population
    :param fit: Fit value for each individual in P
    :param individuals: Number of allowed individuals in the next epoch
    :return: The n most fit individuals
    """
    selected = selection.random_rank_selection(fit, n)
    # Delete temporal data from each non-selected individual
    # 1- Get id of an individual
    # 2- Delete folder in 'FCM/Classifiers/id'
    # 3- Return the best on the array
    new_P = [P[i] for i in selected]
    return new_P


# Plot alive individuals
def plot_population(R, R_old, R_models, iter, time=None):
    f = plt.figure(1)
    plt.clf()
    plt.title("Population iteration %d: " % iter)
    plt.xlabel("Time (s)")
    plt.ylabel("Test accuracy")

    x = [v.test['system'].time for (k, v) in R_models.items()]
    y = [v.test['system'].accuracy for (k, v) in R_models.items()]
    plt.scatter(x, y, label="Initial population", color="black", s=40)

    x = [v.test['system'].time for (k, v) in R_old.items()]
    y = [v.test['system'].accuracy for (k, v) in R_old.items()]
    plt.scatter(x, y, label="Dead individuals", color="red", s=5)

    x = [v.test['system'].time for (k, v) in R.items()]
    y = [v.test['system'].accuracy for (k, v) in R.items()]
    plt.scatter(x, y, label="Alive individuals", color="green", s=10)

    if time is not None:
        plt.axvline(time)

    f.legend()
    plt.show()


def print_stats_population(R):

    front = front_tools.get_front_time_accuracy(R, phase="test")
    front_sorted = front_tools.sort_results_by_accuracy(front, phase="test")
    front_models = front_tools.get_front_time_accuracy(R_models, phase="test")
    front_sorted_models = front_tools.sort_results_by_accuracy(front_models, phase="test")

    print("\t Speedup:%f" % rs.get_speedup_front(front_sorted_models, front_sorted, args.time_constraint, phase="test"))
    print("\t Increment:%f" % rs.get_increment_front(front_sorted_models, front_sorted, args.time_constraint,
                                                     phase="test"))


if __name__ == "__main__":

    global args
    args = argument_parse(sys.argv[1:])
    random.seed()  # Initialize the random generator

    # Initial population
    P = generate_initial_population()
    best_fit = []
    R = {}
    R_old = {}
    R_models = io.read_pickle(os.path.join(os.environ['FCM'], 'Examples', 'pushing_paretto_chain', 'results', 'front45_models_validation', 'models.pkl'))

    # Start the loop over generations
    iteration = 0
    improving = True
    while iteration < args.iterations and improving:
        P = mutation_operations(P)
        # P = breed_operations(P)
        fit = fit_fun.f1_time_penalization(P, time_constraint=args.time_constraint)
        P = select_more_fit(P, fit, args.population)
        iteration += 1

        # Performance evaluation code
        best_fit.append(max(fit))
        print("Iteration %d, best:%f" % (iteration, best_fit[-1]))

        # Plotting population
        R_old.update(R)
        R.clear()
        for p in P:
            R[p.get_sysid()] = ev.evaluate(p, p.get_start(), phases=["test", "val"])
        plot_population(R, R_old, R_models, iteration, args.time_constraint)

        # Speedup and Increment in accuracy with respect to best single model
        print_stats_population(R)

    """
    plt.figure(0)
    plt.plot(best_fit)
    plt.show()
    """

