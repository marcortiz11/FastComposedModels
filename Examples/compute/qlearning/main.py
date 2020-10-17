from Examples.compute.chain_genetic_algorithm.main import generate_initial_population
from Source.genetic_algorithm import operations_mutation as opm
from Source.genetic_algorithm.fitting_functions import f2_time_param_penalization as fit
from Source.genetic_algorithm.fitting_functions import make_limits_dict, update_limit_dict
from Source.io_util import read_pickle
from Source.system_evaluator import evaluate
import Source.make_util as mutil
import Source.system_builder_serializable as sb
import Source.FastComposedModels_pb2 as fcm

from random import random, choice, randint
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys
import time


def take_action(ensemble, Qtable, epsilon, pool_dnns, classifier_path):
    """
    e-Greedy policy for action selection
    :param ensemble: SystemBuilder || Current state, ensemble
    :param Qtable: INT MATRIX || Q(s,a) value table
    :param pool_dnns: LIST STR || Initial pool of individual DNN solutions to the problem
    :param epsilon: FLOAT || epsilon in e-Greedy policy
    :return: Tuple || Action chosen
    """

    if random() < 1-epsilon and \
            max(Qtable.get(ensemble, {}).values(), default=0) > 0:
        return list(Qtable[ensemble].keys())[np.argmax(Qtable[ensemble].values())]
    else:
        total_num_actions = 0
        # Extend Chain actions
        tail_chains = [c.id for c in ensemble.get_message().classifier if c.component_id == ""]
        a = len(pool_dnns) * len(tail_chains)
        # Increase/Decrease threshold actions
        non_tail = [c.id for c in ensemble.get_message().classifier if c.component_id != ""]
        b = len(non_tail)
        total_num_actions += a + 2 * b
        c = d = 0

        if len(ensemble.get_message().merger) > 0:
            # Add classifier to the merger actions
            c = len(pool_dnns)
            # Change merging protocol actions
            merger_id = list(ensemble.get_message().merger)[0].id
            merging_protocols = list(fcm.Merger.MergeType)
            d = len(merging_protocols)
            total_num_actions += c + d

        action = None
        r = random() * total_num_actions
        if r < a:  # Extend chain
            c_id = choice(pool_dnns)
            c_file = os.path.join(classifier_path, c_id)
            action = (opm.extend_merged_chain, choice(tail_chains), c_id, 0.5, c_file)
        if a < r < a+b:  # Increase threshold
            action = (opm.update_threshold, choice(non_tail), +0.1)
        if a+b < r < a+2*b:  # Decrease threshold
            action = (opm.update_threshold, choice(non_tail), -0.1)
        if a+2*b < r < a+2*b+c:  # Add one more classifier to the merger
            c_id = choice(pool_dnns)
            c_file = os.path.join(classifier_path, c_id)
            action = (opm.add_classifier_to_merger, merger_id, c_id, c_file)
        if r > a+2*b+c:  # Change merging protocol
            action = (opm.change_merging_protocol, merger_id, choice(merging_protocols))

        return action


def argument_parse(argv):
    parser = argparse.ArgumentParser(usage="main.py [options]",
                                     description="Optimal Ensemble Generation Policy with Q-Learning")
    parser.add_argument("--dataset", default="sota_models_cifar10-32-dev_validation", help="Datasets to evaluate d1 d2 d3, default=*")
    # Search-space params
    parser.add_argument("--episodes", default=10000, type=int, help="Number of episodes")
    parser.add_argument("--steps", default=40, type=int, help="Number of actions explored in each episode")
    parser.add_argument("--alpha", default=0.2, type=float, help="Q-Learning's learning rate parameter")
    parser.add_argument("--gamma", default=0.99, type=float, help="Long-Term reward gamma factor")
    parser.add_argument("--e", type=float, default=0.25)
    parser.add_argument("--w", nargs='+', default=[0.5, 0.5, 0], type=float, help="Fitting function's weight 0..1 where [acc, time, params]")
    # Execution parameters
    parser.add_argument("--plot", default=0, type=int, help="Plot episode reward")
    parser.add_argument("--comment", default="", type=str, help="")

    return parser.parse_args(argv)


class ProgressBar:
    def __init__(self, steps):
        self.steps = steps
        self.i = 0

    def __str__(self):
        self.i += 1
        return "[[%s]]" % ("#"*(self.i%self.steps) + " "*(self.steps-self.i%self.steps))


if __name__ == "__main__":

    args = argument_parse(sys.argv[1:])
    os.environ['TMP'] = 'Definitions/Classifiers/tmp/' + args.dataset[12:-15]

    # Load 32 DNNs
    S_initial = []
    S_eval_dict = {}
    limits = make_limits_dict()

    classifier_path = os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers', args.dataset)
    classifier_files = [f for f in os.listdir(classifier_path) if ".pkl" in f]
    for c_id in classifier_files:
        sys = sb.SystemBuilder(verbose=False)
        c_file = os.path.join(classifier_path, c_id)
        sys.add_classifier(mutil.make_classifier(c_id, c_file))
        sys.set_start(c_id)
        S_initial.append(sys)
        S_eval_dict[c_id] = evaluate(sys, sys.get_start(), phases=["val"])
    update_limit_dict(limits, S_eval_dict, phase="val")

    # Initialize Q-Learning table
    Qtable = {}

    # Start Q-loop
    bar = ProgressBar(args.steps)
    R_episodes = []
    for episode in range(args.episodes):
        print("EPISODE %d" % episode)
        ensemble = S_initial[(episode%len(S_initial))]
        ensemble_eval = evaluate(ensemble, ensemble.get_start(), phases=["val"])
        R = 0   # Cumulative episode reward

        t_start = time.time()
        for step in range(args.steps):
            print(bar, end="\r")

            # Take an action
            a = take_action(ensemble, Qtable, args.e, classifier_files, classifier_path)

            # Modify ensemble with action
            ensemble_new = ensemble.copy()
            a[0](ensemble_new, *a[1:])

            # Evaluate ensemble
            ensemble_eval_new = evaluate(ensemble_new, ensemble_new.get_start(), phases=["val"])
            r = fit([ensemble_eval_new], args.w, limits, phase="val")[0] - \
                fit([ensemble_eval], args.w, limits, phase="val")[0]

            # Update Q-Table
            if not Qtable.get(ensemble):
                Qtable[ensemble] = {}
            qval = Qtable[ensemble].get(a, 0.0)
            qval_next = max(Qtable.get(ensemble_new, {}).values(), default=0)
            Qtable[ensemble][a] = qval + args.alpha*(r + args.gamma*qval_next - qval)

            # Next state
            ensemble = ensemble_new
            ensemble_eval = ensemble_eval_new
            R += pow(args.gamma, step)*r

        R_episodes.append(R)
        print("\t Episode reward: %f" % R)
        print("\t Episode elapsed time: %f sec" % (time.time() - t_start))
        print()

    # Plot evolution
    if args.plot:
        plt.figure()
        plt.title("Episode reward evolution")
        plt.xlabel("Episode")
        plt.ylabel("Episode reward R")
        plt.plot(list(range(args.episodes)), R_episodes)
        plt.show()

    # Save Qtable and Episode rewards evolution
    import Examples.metadata_manager_results as manager_results

    meta_data_file = os.path.join(os.environ['FCM'],
                                  'Examples',
                                  'compute',
                                  'qlearning',
                                  'results',
                                  'metadata.json')

    id = str(randint(0, 1e16))
    params = args.__dict__
    results_loc = os.path.join('Examples/compute/qlearning/results', args.dataset, id)
    comments = args.comment
    meta_data_result = manager_results.metadata_template(id, args.dataset, results_loc, comments)
    manager_results.save(meta_data_file, meta_data_result, params, ("reward_evolution", R_episodes), ("qtable", Qtable))