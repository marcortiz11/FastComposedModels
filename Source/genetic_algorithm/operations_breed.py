from Source.genetic_algorithm.operations_mutation import __make_data_and_dataset
import Source.make_util as make
import copy
import os


#---- VERSION 1: Find the best chain ----#
def remove(a, point, first=True):
    """
    :param a: system
    :param point: point is an id of the component in the system A
    :return: The system dropping all the elements following that id
    """
    component = a.get(point)
    if component.DESCRIPTOR.name == "Trigger":
        next = component.component_ids
        for n in next:
            remove(a, n, False)  # Removing connected components before
        # Delete data definitions
        if component.classifier.data_id != '':
            a.remove(component.classifier.data_id)
    elif component.DESCRIPTOR.name == "Classifier":
        next = component.component_id
        if next != '':
            remove(a, next, False)  # Removing connected components before

    # Finally remove myself
    if not first:
        a.remove(point)
    else:
        component.component_id = ''  # There's nothing after


def merge_chains_by_point(a, b, pointA, pointB):
    """
    :param a: Chain to be added part of chain b
    :param b: Chain b
    :param point: Point in the chain b form which to start adding
    :return: New chain
    """

    next = [b.get(pointB).component_id] if b.get(pointB).component_id != '' else []  # Always should be a Classifier

    if len(next) > 0:
        th = float(next[0].split("_")[2])
        trigger_name = "trigger_classifier_" + str(th) + "_" + pointA
        trigger_classifier_file = os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers', 'tmp',
                                               trigger_name + '.pkl')

        if not os.path.exists(trigger_classifier_file):
            data = __make_data_and_dataset(a, pointA, a.get(pointA).classifier_file, th)
            a.add_data(data)  # TODO: Més eficient!
            trigger = make.make_trigger(trigger_name,
                                        make.make_empty_classifier(data_id=data.id),
                                        component_ids=b.get(next[0]).component_ids, model="probability")
        else:
            trigger = make.make_trigger(trigger_name,
                                        make.make_classifier('', classifier_file=trigger_classifier_file),
                                        component_ids=b.get(next[0]).component_ids)

        a.add_trigger(trigger)
        a.get(pointA).component_id = trigger_name
        next = trigger.component_ids

    while len(next) > 0:
        assert len(next) == 1, "ERROR: Support only for chain ensemble with crossover operations"

        next_id = next[0]
        c = b.get(next_id)
        c_ = copy.deepcopy(c)

        if c.DESCRIPTOR.name == "Classifier":
            next = [c.component_id] if c.component_id != '' else []
            a.add_classifier(c_)

        elif c.DESCRIPTOR.name == "Trigger":
            next = c.component_ids
            a.add_trigger(c_)


def repeated_classifiers(a, b, pointB):
    classifiers = [classifier.id for classifier in a.get_message().classifier]
    next = [b.get(pointB).component_id] if b.get(pointB).component_id != '' else []  # Always should be a Classifier

    while len(next) > 0:
        assert len(next) == 1, "ERROR: Support only for chain ensemble with crossover operations"
        next_id = next[0]
        c = b.get(next_id)
        if c.DESCRIPTOR.name == "Classifier":
            if c.id in classifiers:
                return True
            next = [c.component_id] if c.component_id != '' else []
        elif c.DESCRIPTOR.name == "Trigger":
            next = c.component_ids  # El classificador del trigger ha d'estar entrenat

    return False


def singlepoint_crossover(a, b, pointA, pointB):
    """
    Reproduction of individuals a and b using the single point crossover operation
    :param a: Individual (ensemble model)
    :param b: Individual (ensemble model)
    :param point: Crossover point (Trigger or Classifier id)
    :return: Two new individuals, the child of a and b.
    """

    assert a.get(pointA).DESCRIPTOR.name == "Classifier" and b.get(pointB).DESCRIPTOR.name == "Classifier", \
        "Crossover points A and B should point to Classifiers"


    offspring = []

    c1 = a.copy()
    c2 = b.copy()

    remove(c1, pointA)
    remove(c2, pointB)

    # Only return plausible offspring (do not create offspring chains that have same classifier in the chain)
    if not repeated_classifiers(c1, b, pointB):
        merge_chains_by_point(c1, b, pointA, pointB)
        offspring += [c1]
    if not repeated_classifiers(c2, a, pointA):
        merge_chains_by_point(c2, a, pointB, pointA)
        offspring += [c2]

    return offspring
