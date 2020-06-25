import Source.FastComposedModels_pb2 as fcm
import Source.make_util as make
import Source.io_util as io
import os


def __update_dataset(c_file, train_path, test_path, val_path, th):
    # Create dataset
    model = io.read_pickle(c_file)
    # Test
    dataset_test = model['test']
    dataset_test['th'] = th
    io.save_pickle(test_path, dataset_test)
    # Train
    """
    dataset_train = model['train']
    dataset_train['th'] = th
    io.save_pickle(train_path, dataset_train)
    """
    # Validation
    dataset_val = model['val']
    dataset_val['th'] = th
    io.save_pickle(val_path, dataset_val)


def __make_data_and_dataset(i, c_id, c_file, th):

    individual_id = i.get_sysid()
    number_individual_id = str(int.from_bytes(individual_id.encode('utf-8'), byteorder='big'))

    # Create dataset for new trigger
    train_path = os.path.join(os.environ['FCM'], 'Data', 'DATASET_' + str(number_individual_id[:10]) + '_' + c_id + '_train')
    test_path = os.path.join(os.environ['FCM'], 'Data', 'DATASET_' + str(number_individual_id[:10]) + '_' + c_id + '_test')
    val_path = os.path.join(os.environ['FCM'], 'Data', 'DATASET_' + str(number_individual_id[:10]) + '_' + c_id + '_val')
    __update_dataset(c_file, train_path, test_path, val_path, th)

    # Define data
    data_id = 'data_' + number_individual_id + '_' + c_id
    source = make.make_source(train_path, test_path, 2, val_path)
    data = make.make_data(data_id, int(5e4), int(5e3), None, source)

    return data

#################################################
#   ---- VERSION 1: Find the best chain ----    #
#################################################


def extend_chain_pt(i, c_id, th=None, c_file = None, trigger_name=None):

    """
    Extends the chain with a probability trigger and a classifier
    :param i: Ensemble of models (system) representing an individual
    :param c_id: Classifier to attach at the end fo the chain
    :param th: Threshold of the trigger
    :return: Nothing
    """

    th = 0.5 if th is None else th
    c_file = c_id if c_file is None else c_file
    existing_classifier_id = [c.id for c in i.get_message().classifier]

    if c_id not in existing_classifier_id:  # Add classifier if not present previously in the chain

        # Get the last model in the chain
        for c in i.get_message().classifier:
            if c.component_id == "":
                last_id = c.id

        # Create dataset for new trigger if trigger's classifier not existing
        if trigger_name is None:
            trigger_name = "trigger_classifier_" + str(th) + "_" + last_id
        trigger_classifier_file = os.path.join(os.environ['FCM'], os.environ['TMP'], trigger_name+'.pkl')

        if not os.path.exists(trigger_classifier_file):
            data = __make_data_and_dataset(i, last_id, i.get(last_id).classifier_file, th)
            i.add_data(data)
            trigger = make.make_trigger(trigger_name, make.make_empty_classifier(id="", data_id=data.id), component_ids=[c_id], model="probability")
        else:
            trigger = make.make_trigger(trigger_name, make.make_classifier('', trigger_classifier_file), component_ids=[c_id])

        i.add_trigger(trigger)

        # Create new mutated classifier
        classifier = make.make_classifier(c_id, c_file)
        i.add_classifier(classifier)

        # Update last classifier to connect to trigger
        last_classifier = i.get(last_id)
        last_classifier.component_id = trigger_name
        i.replace(last_id, last_classifier)


def replace_classifier(i, c_id, c_id_new, c_file = None, trigger_name = None):

    """
    Replaces a classifier in the chain for a new classifier
    :param i: Ensemble of models (system) representing an individual
    :param c_id: Classifier's id to be replaced
    :param c_id_new: Classifier's id that replaces
    :return: Nothing
    """

    c_file = c_id if c_file is None else c_file
    existing_classifier_id = [c.id for c in i.get_message().classifier]

    if c_id_new not in existing_classifier_id:
        classifier = make.make_classifier(c_id_new, c_file)
        if i.get(c_id).component_id != '':

            # Check it is a trigger
            trigger_id = i.get(c_id).component_id
            assert i.get(trigger_id).DESCRIPTOR.name == "Trigger", \
                "ERROR: Classifier in chain should be connected to trigger"

            # Replace previous data component on the system for the new one
            trigger_classifier_old = i.get(trigger_id).classifier
            old_data_id = trigger_classifier_old.data_id
            th = float(trigger_id.split("_")[2])

            if trigger_name is None:
                trigger_name = "trigger_classifier_" + str(th) + "_" + c_id_new
            trigger_classifier_file = os.path.join(os.environ['FCM'], os.environ['TMP'], trigger_name+'.pkl')

            if not os.path.exists(trigger_classifier_file):
                data = __make_data_and_dataset(i, c_id_new, c_file, th)
                i.replace(old_data_id, data)
                trigger = make.make_trigger(trigger_name, make.make_empty_classifier(id="", data_id=data.id),
                                            component_ids=i.get(trigger_id).component_ids, model="probability")
            else:
                trigger = make.make_trigger(trigger_name,
                                            make.make_classifier('', classifier_file=trigger_classifier_file),
                                            component_ids=i.get(trigger_id).component_ids)
            i.replace(trigger_id, trigger)

            # Link replacing classifier to trigger
            classifier.component_id = trigger_name

        # Replace classifier
        i.replace(c_id, classifier)

        # If first classifier, point at it
        if i.get_start() == c_id:
            i.set_start(c_id_new)

        # Get trigger connected to the old classifier
        trigger_names = i.get_message().trigger  # All triggers from the ensemble
        for trigger in trigger_names:
            trigger = i.get(trigger.id)
            for ic, c in enumerate(trigger.component_ids):
                if c_id == c:
                    trigger.component_ids[ic] = c_id_new
                    i.replace(trigger.id, trigger)


def update_threshold(i, c_id, step, trigger_name=None):
    """
    Updates the threshold of a classifier by adding step (+ or -)
    :param i: Ensemble of models (system) representing an individual
    :param c_id: Classifier affected
    :param step: Value to which increment or decrement the threshold
    :return: Nothing
    """

    c_file = i.get(c_id).classifier_file
    if i.get(c_id).component_id != '':

        trigger_id = i.get(c_id).component_id
        trigger_old = i.get(trigger_id)
        assert trigger_old.DESCRIPTOR.name == "Trigger", "Classifiers should be connected to triggers in the chain"

        th = float(trigger_id.split("_")[2])
        new_th = th+step

        if trigger_name is None:
            trigger_name = "trigger_classifier_" + str(new_th) + "_" + c_id
        trigger_classifier_file = os.path.join(os.environ['FCM'], os.environ['TMP'], trigger_name+'.pkl')

        if not os.path.exists(trigger_classifier_file):
            data = __make_data_and_dataset(i, c_id, c_file, new_th)
            i.replace(trigger_old.classifier.data_id, data)
            trigger = make.make_trigger(trigger_name,
                                        make.make_empty_classifier(id="", data_id=data.id),
                                        component_ids=i.get(trigger_id).component_ids, model="probability")
        else:
            trigger = make.make_trigger(trigger_name,
                                        make.make_classifier('', classifier_file=trigger_classifier_file),
                                        component_ids=i.get(trigger_id).component_ids)
        i.replace(trigger_old.id, trigger)
        c = i.get(c_id)
        c.component_id = trigger_name
        i.replace(c_id, c)


############################################################
#   ---- VERSION 2: Bagging and Boosting of Chains ----    #
############################################################

def add_classifier_to_merger(i, merger_id, classifier_id, classifier_file):

    # Add the new classifier to the system
    classifier = make.make_classifier(classifier_id, classifier_file)
    i.add_classifier(classifier)

    # Extend the merger
    merger = i.get(merger_id)
    merger.merged_ids.append(classifier_id)
    i.replace(merger_id, merger)


def change_merging_protocol(i, merger_id, merge_protocol=fcm.Merger.AVERAGE):

    # Change the merging protocol
    merger = i.get(merger_id)
    merger.merge_type = merge_protocol
    i.replace(merger_id, merger)


def extend_merged_chain(i, c_id_tail, c_id_new, th=None, c_file_new=None, trigger_name=None):

    """

    :param i: Individual
    :param c_id_tail: Last classifier at the chain
    :param c_id_new: Id of new last classifier at the chain
    :param th: Threshold value of the new trigger
    :param c_file_new: File location of the classifier at the chain
    :param t_id: Id of the new trigger
    :return: Nothing
    """

    c_file = c_id_tail if c_file_new is None else c_file_new
    existing_classifier_id = [c.id for c in i.get_message().classifier]

    if c_id_new not in existing_classifier_id:  # Add classifier if not present previously in the chain
        if trigger_name is None:
            trigger_name = "trigger_classifier_" + str(th) + "_" + c_id_tail
        trigger_classifier_file = os.path.join(os.environ['FCM'], os.environ['TMP'], trigger_name+'.pkl')

        if not os.path.exists(trigger_classifier_file):
            data = __make_data_and_dataset(i, c_id_tail, i.get(c_id_tail).classifier_file, th)
            i.add_data(data)
            trigger = make.make_trigger(trigger_name, make.make_empty_classifier(id="", data_id=data.id), component_ids=[c_id_new], model="probability")
        else:
            trigger = make.make_trigger(trigger_name, make.make_classifier('', trigger_classifier_file), component_ids=[c_id_new])

        i.add_trigger(trigger)

        # Create new mutated classifier
        classifier = make.make_classifier(c_id_new, c_file)
        i.add_classifier(classifier)

        # Update last classifier to connect to trigger
        last_classifier = i.get(c_id_tail)
        last_classifier.component_id = trigger_name
        i.replace(c_id_tail, last_classifier)


def replace_classifier_merger(i, c_id, c_id_new, c_file=None, trigger_name=None):

        """
        Replaces a classifier in the chain for a new classifier
        :param i: Ensemble of models (system) representing an individual
        :param c_id: Classifier's id to be replaced
        :param c_id_new: Classifier's id that replaces
        :return: Nothing
        """

        c_file = c_id if c_file is None else c_file
        existing_classifier_id = [c.id for c in i.get_message().classifier]

        if c_id_new not in existing_classifier_id:
            classifier = make.make_classifier(c_id_new, c_file)
            if i.get(c_id).component_id != '':

                # Check it is a trigger
                trigger_id = i.get(c_id).component_id
                assert i.get(trigger_id).DESCRIPTOR.name == "Trigger", \
                    "ERROR: Classifier in chain should be connected to trigger"

                # Replace previous data component on the system for the new one
                trigger_classifier_old = i.get(trigger_id).classifier
                old_data_id = trigger_classifier_old.data_id
                th = float(trigger_id.split("_")[2])

                if trigger_name is None:
                    trigger_name = "trigger_classifier_" + str(th) + "_" + c_id_new
                trigger_classifier_file = os.path.join(os.environ['FCM'], os.environ['TMP'], trigger_name + '.pkl')

                if not os.path.exists(trigger_classifier_file):
                    data = __make_data_and_dataset(i, c_id_new, c_file, th)
                    i.replace(old_data_id, data)
                    trigger = make.make_trigger(trigger_name, make.make_empty_classifier(id="", data_id=data.id),
                                                component_ids=i.get(trigger_id).component_ids, model="probability")
                else:
                    trigger = make.make_trigger(trigger_name,
                                                make.make_classifier('', classifier_file=trigger_classifier_file),
                                                component_ids=i.get(trigger_id).component_ids)
                i.replace(trigger_id, trigger)

                # Link replacing classifier to trigger
                classifier.component_id = trigger_name

            # Replace classifier
            i.replace(c_id, classifier)

            # If first classifier, point at it
            if i.get_start() == c_id:
                i.set_start(c_id_new)

            # Update the merger if there's any
            for merger in i.get_message().merger:
                for idx, merged_id in enumerate(merger.merged_ids):
                    if merged_id == c_id:
                        merger.merged_ids[idx] = c_id_new
                        i.replace(merger.id, merger)

            # Get trigger connected to the old classifier
            trigger_names = i.get_message().trigger  # All triggers from the ensemble
            for trigger in trigger_names:
                trigger = i.get(trigger.id)
                for ic, c in enumerate(trigger.component_ids):
                    if c_id == c:
                        trigger.component_ids[ic] = c_id_new
                        i.replace(trigger.id, trigger)
