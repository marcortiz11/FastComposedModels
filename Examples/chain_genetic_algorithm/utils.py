import matplotlib.pyplot as plt
import Examples.paretto_front as front_tools
import Examples.fully_connected_chain.results_study as rs
import os, random


def pick_random_threshold(args):
    return 0.5


def pick_random_classifier(args, sys=None):
    classifier_path = os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers', args.datasets)
    classifier_files = [os.path.join(classifier_path, f) for f in os.listdir(classifier_path) if ".pkl" in f]
    selected = classifier_files[random.randint(0, len(classifier_files)-1)]
    # Pick only classifiers with higher accuracy
    if sys is not None:
        classifier_files = sys.get_message().classifier
        selected = classifier_files[random.randint(0, len(classifier_files)-1)].id
    return selected


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
    # plt.pause(0.1)
    plt.show()


def print_stats_population(R, R_models, args):

    front = front_tools.get_front_time_accuracy(R, phase="test")
    front_sorted = front_tools.sort_results_by_accuracy(front, phase="test")
    front_models = front_tools.get_front_time_accuracy(R_models, phase="test")
    front_sorted_models = front_tools.sort_results_by_accuracy(front_models, phase="test")

    print("\t Speedup:%f" % rs.get_speedup_front(front_sorted_models, front_sorted, args.time_constraint, phase="test"))
    print("\t Increment:%f" % rs.get_increment_front(front_sorted_models, front_sorted, args.time_constraint,
                                                     phase="test"))
