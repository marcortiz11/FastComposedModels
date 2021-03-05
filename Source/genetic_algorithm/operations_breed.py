from Source.genetic_algorithm.operations_mutation import __make_data_and_dataset
import Source.protobuf.make_util as make
from Examples.compute.chain_genetic_algorithm.utils import get_classifier_index
import copy
import os


#################################################
#   ---- VERSION 1: Find the best chain ----    #
#################################################

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
        # Delete data Definitions
        if component.classifier.data_id != '':
            a.remove(component.classifier.data_id)
    elif component.DESCRIPTOR.name == "Classifier":
        next = component.component_id
        if next != '':
            remove(a, next, False)  # Removing connected components before

    # Finally remove myself
    a.remove(point)


"""
def merge_chains_by_point(a, b, pointA, pointB):
    
    :param a: Chain to be added part of chain b
    :param b: Chain b
    :param point: Point in the chain b form which to start adding
    :param sub_folder: Folder where trained trigger's classifiers are saved
    :return: New chain
    

    next = [b.get(pointB).component_id] if b.get(pointB).component_id != '' else []
    merge_level = pointA[0:2] if str.isdecimal(pointA[0]) else ''

    if len(next) > 0:
        th = float(next[0].split("_")[2])
        trigger_name = "trigger_classifier_" + str(th) + "_" + pointA
        trigger_classifier_file = os.path.join(os.environ['FCM'], os.environ['TMP'], trigger_name + '.pkl')
        trigger_c_ids = [merge_level+id[2:] if str.isdecimal(id[0]) else id for id in b.get(next[0]).component_ids]

        if not os.path.exists(trigger_classifier_file):
            data = __make_data_and_dataset(a, pointA, a.get(pointA).classifier_file, th)
            a.add_data(data)  # TODO: Més eficient!
            trigger = make.make_trigger(trigger_name,
                                        make.make_empty_classifier(id="", data_id=data.id),
                                        component_ids=trigger_c_ids, model="probability")
        else:
            trigger = make.make_trigger(trigger_name,
                                        make.make_classifier('', classifier_file=trigger_classifier_file),
                                        component_ids=trigger_c_ids)

        a.add_trigger(trigger)
        c = a.get(pointA)
        c.component_id = trigger_name
        a.replace(c.id, c)
        next = b.get(next[0]).component_ids

    while len(next) > 0:
        assert len(next) == 1, "ERROR: Support only for chain ensemble with crossover operations"

        next_id = next[0]
        c = b.get(next_id)
        c_ = copy.deepcopy(c)

        if c.DESCRIPTOR.name == "Classifier":
            next = [c.component_id] if c.component_id != '' else []
            c_.id = merge_level+c_.id[2:] if str.isdecimal(c_.id[0]) else c_.id
            if c_.component_id != '':
                c_.component_id = merge_level+c_.component_id[2:] if str.isdecimal(c_.component_id[0]) else c_.component_id
            a.add_classifier(c_)

        elif c.DESCRIPTOR.name == "Trigger":
            next = c.component_ids
            c_.id = merge_level+c_.id[2:] if str.isdecimal(c_.id[0]) else c_.id
            c_.component_ids[:] = [merge_level+id[2:] if str.isdecimal(id[0]) else id for id in c_.component_ids]
            a.add_trigger(c_)
"""


def merge_chains_by_point_v2(a, b, pointA, pointB):
    """
    :param a: Chain to be added part of chain b
    :param b: Chain b
    :param point: Point in the chain b form which to start adding
    :param sub_folder: Folder where trained trigger's classifiers are saved
    :return: New chain
    """

    # Remove classifier
    next = [pointB]
    merge_level = pointA[0:2] if str.isdecimal(pointA[0]) else ''
    a.remove(pointA)

    # If point A was starter, set pointB as starter
    if a.get_start() == pointA:
        a.set_start(merge_level + (pointB[2:] if str.isdecimal(pointB[0]) else pointB))

    # If is head of chain in merger change id to pointB
    mergers = a.get_message().merger
    for merger in mergers:
       for idx, id in enumerate(merger.merged_ids):
            if id == pointA:
                merger.merged_ids[idx] = merge_level + (pointB[2:] if str.isdecimal(pointB[0]) else pointB)
                a.replace(merger.id, merger)
                break

    # Trigger connected to classifier=pointA should be reconnected to point B
    triggers = a.get_message().trigger
    for trigger in triggers:
        if pointA in trigger.component_ids:
            trigger_ = copy.deepcopy(trigger)
            trigger_.component_ids[0] = merge_level + (pointB[2:] if str.isdecimal(pointB[0]) else pointB)
            a.replace(trigger.id, trigger_)
            break

    # Copy the tail of ensemble b from pointB
    while len(next) > 0:
        assert len(next) == 1, "ERROR: Support only for chain ensemble with crossover operations"

        next_id = next[0]
        c = b.get(next_id)
        c_ = copy.deepcopy(c)

        if c.DESCRIPTOR.name == "Classifier":
            next = [c.component_id] if c.component_id != '' else []
            c_.id = merge_level + (c_.id[2:] if str.isdecimal(c_.id[0]) else c_.id)
            if c_.component_id != '':
                aux = '_'.join(c_.component_id.split("_")[3:])
                c_.component_id = "trigger_classifier_" + c_.component_id.split("_")[2] + "_" + merge_level + (aux[2:] if str.isdecimal(aux[0]) else aux)
            a.add_classifier(c_)

        elif c.DESCRIPTOR.name == "Trigger":
            next = c.component_ids
            c_.component_ids[:] = [merge_level + (id[2:] if str.isdecimal(id[0]) else id) for id in c_.component_ids]
            aux = '_'.join(c_.id.split("_")[3:])
            c_.id = "trigger_classifier_" + c_.id.split("_")[2] + "_" + merge_level + (aux[2:] if str.isdecimal(aux[0]) else aux)
            a.add_trigger(c_)


def repeated_classifiers(a, b, pointB, level):

    # Point to the head of the chain in the ensemble
    level = level[0:2] if str.isdecimal(level[0]) and level[1] == '_' else ""
    if len(a.get_message().merger) > 0:
        merger = a.get_message().merger[0]
        for m_id in merger.merged_ids:
            if m_id[0:2] == level or (level == "" and not str.isdecimal(m_id[0]) and not m_id[1] == "_"):
                start_chain = m_id
                break
    else:
        start_chain = a.get_start()

    # Store all classifiers in the non-replaced part of the chain
    i = 0
    classifiers = []
    while get_classifier_index(a, i, point=start_chain) is not None:
        classifiers.append(get_classifier_index(a, i, start_chain))
        i+=1
    classifiers += [classifier[2:] for classifier in classifiers]

    # Look for repetitions in classifiers when concatenating the two parts of chain a and chain b
    next = [pointB]
    while len(next) > 0:
        assert len(next) == 1, "ERROR: Support only for chain ensemble with crossover operations"
        next_id = next[0]
        c = b.get(next_id)
        if c.DESCRIPTOR.name == "Classifier":
            if c.id in classifiers or c.id[2:] in classifiers:
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
    if not repeated_classifiers(c1, b, pointB, pointA):
        merge_chains_by_point_v2(c1, b, pointA, pointB)
        offspring += [c1]
    if not repeated_classifiers(c2, a, pointA, pointB):
        merge_chains_by_point_v2(c2, a, pointB, pointA)
        offspring += [c2]

    return offspring


############################################################
#   ---- VERSION 2: Bagging and Boosting of Chains ----    #
############################################################

def merge_two_chains(a, b):
    """
    :param a: Chain of classifiers
    :param b: Chain of classifiers
    :return: Merged chains a and b
    """
    m_system = a.copy()

    # 1) Get b protobuf structure
    b_proto_mesage = b.get_message()

    # 2) For each component (Trigger, Data, Classifier) add to m_system with different name
    for classifier in b_proto_mesage.classifier:
        classifier_ = copy.deepcopy(classifier)
        classifier_.id = "1_"+classifier_.id
        classifier_.component_id = "_".join(classifier_.component_id.split('_')[0:3] + ['1'] + classifier_.component_id.split(
            '_')[3:]) if classifier_.component_id != "" else ""
        classifier_.data_id = "1_" + classifier_.data_id if classifier_.data_id != "" else ""
        m_system.add_classifier(classifier_)

    for trigger in b_proto_mesage.trigger:
        trigger_ = copy.deepcopy(trigger)
        trigger_.id = '_'.join(trigger_.id.split('_')[0:3]+['1']+trigger_.id.split('_')[3:])
        for i in range(len(trigger_.component_ids)):
            trigger_.component_ids[i] = "1_" + trigger_.component_ids[i]
        trigger_.classifier.data_id = "1_" + trigger_.classifier.data_id if trigger_.classifier.data_id != "" else ""
        m_system.add_trigger(trigger_)

    for data in b_proto_mesage.data:
        data_ = copy.deepcopy(data)
        data_.id = "1_" + data.id
        m_system.add_data(data_)

    # 3) Finally add merger on m_system
    import Source.protobuf.FastComposedModels_pb2 as fcm
    merger = make.make_merger('Merger', [a.get_start(), "1_"+b.get_start()],
                              merge_type=fcm.Merger.AVERAGE)
    m_system.set_start('Merger')
    m_system.add_merger(merger)

    return m_system







