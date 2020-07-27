import matplotlib.pyplot as plt
import Examples.study.paretto_front as front_tools
import Examples.compute.fully_connected_chain.results_study as rs
import os, random


def generate_system_id(system):
    component = system.get(system.get_start())
    id = ""
    if component.DESCRIPTOR.name == "Merger":
        id += generate_system_id_merger(system, component)
    else:
        id += generate_system_id_chain(system, component)
    return id


def generate_system_id_merger(system, component, depth=1):
    """
    Generates an id for the individuals of the genetic algorithm. Useful for detecting duplicates,
    and creating readable ids for individuals.
    :param chain: An ensemble
    :return: String id
    """
    id = "Merge: "
    merged = component.merged_ids
    for c_id in merged:
        c = system.get(c_id)
        id += "\n|" + "\t"*depth
        if c.DESCRIPTOR.name == "Merger":
            id += generate_system_id_merger(system, c, depth+1)
        else:
            id += generate_system_id_chain(system, c)
    return id


def generate_system_id_chain(system, c):
    """
    Generates an id for the individuals of the genetic algorithm. Useful for detecting duplicates,
    and creating readable ids for individuals.
    :param chain: An ensemble
    :return: String id
    """
    id = ""
    component = c
    while component is not None:
        if component.DESCRIPTOR.name == "Trigger":
            next_chain = component.component_ids
            assert len(next_chain) < 3, "ERROR: gen_system_id_chain only works with chains"
            id += "__%s__" % component.id
            component = system.get(next_chain[0])
        else:  # Classifier
            next = component.component_id
            if next != '':
                component = system.get(next)
            else:
                id += "%s" % component.id
                component = None
    return id


def pick_random_threshold(args):
    return 0.5


def pick_random_classifier(args, sys=None):
    classifier_path = os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers', args.dataset)
    classifier_files = [os.path.join(classifier_path, f) for f in os.listdir(classifier_path) if ".pkl" in f]
    selected = classifier_files[random.randint(0, len(classifier_files)-1)]
    # Pick only classifiers with higher accuracy
    if sys is not None:
        classifier_files = sys.get_message().classifier
        selected = classifier_files[random.randint(0, len(classifier_files)-1)].id
    return selected


def get_classifier_index(chain, i):

    """
    :param chain: Ensemble chain
    :param i: index of the classifier in the chain
    :return: The classifier id in the index i
    """

    assert len(chain.get_message().classifier) > i, "ERROR: Index out of bounds"

    c = chain.get(chain.get_start())
    next = c.component_id
    i_ = 0
    while i_ != i:
        trigger = chain.get(next)
        if trigger is None:
            break
        c = chain.get(trigger.component_ids[0])
        next = c.component_id
        i_ += 1
    return c.id


# Plot alive individuals
def plot_population(R, R_models, iter, fit=None):
    f = plt.figure(1)
    plt.clf()

    plt.title("Population iteration %d: " % iter)
    plt.xlabel("Time (s)")
    plt.ylabel("Test accuracy")

    x = [v.test['system'].time for (k, v) in R_models.items()]
    y = [v.test['system'].accuracy for (k, v) in R_models.items()]
    plt.scatter(x, y, label="Initial population", color="black", s=40)
    plt.hlines(max(y), 0, max(x), color="red")

    x = [v.test['system'].time for (k, v) in R.items()]
    y = [v.test['system'].accuracy for (k, v) in R.items()]

    if fit is not None:
        plt.scatter(x, y, c=fit, label="Alive individuals", s=10)
    else:
        plt.scatter(x, y, label="Alive individuals", color="green", s=10)

    f.legend()
    plt.pause(0.1)
    # plt.show()


def print_stats_population(R, R_models, args):

    front = front_tools.get_front_time_accuracy(R, phase="test")
    front_sorted = front_tools.sort_results_by_accuracy(front, phase="test")
    front_models = front_tools.get_front_time_accuracy(R_models, phase="test")
    front_sorted_models = front_tools.sort_results_by_accuracy(front_models, phase="test")

    print("\t Speedup:%f" % rs.get_speedup_front(front_sorted_models, front_sorted, args.time_constraint, phase="test"))
    print("\t Increment:%f" % rs.get_increment_front(front_sorted_models, front_sorted, args.time_constraint,
                                                     phase="test"))
