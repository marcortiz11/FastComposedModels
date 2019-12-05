import Source.make_util as make
import Source.io_util as io
import os


def __remove_dataset(train_path, test_path, val_path):
    if os.path.exists(train_path+'.pkl'):
        os.remove(train_path+'.pkl')
    if os.path.exists(test_path+'.pkl'):
        os.remove(test_path+'.pkl')
    if os.path.exists(val_path+'.pkl'):
        os.remove(val_path+'.pkl')


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

    # Create dataset for new trigger
    train_path = os.path.join(os.environ['FCM'], 'Data', 'DATASET_' + str(individual_id) + '_' + c_id + '_train')
    test_path = os.path.join(os.environ['FCM'], 'Data', 'DATASET_' + str(individual_id) + '_' + c_id + '_test')
    val_path = os.path.join(os.environ['FCM'], 'Data', 'DATASET_' + str(individual_id) + '_' + c_id + '_val')
    __update_dataset(c_file, train_path, test_path, val_path, th)

    # Define data
    data_id = 'data_' + individual_id + '_' + c_id
    source = make.make_source(train_path, test_path, 2, val_path)
    data = make.make_data(data_id, int(5e4), int(5e3), None, source)

    return data


#   ---- VERSION 1: Find the best chain ----    #
def extend_chain_pt(i, c_id, th=None, c_file = None):

    """
    Extends the chain with a probability trigger and a classifier
    :param i: Ensemble of models (system) representing an individual
    :param c_id: Classifier to attach at the end fo the chain
    :param th: Threshold of the trigger
    :return: A mutated individual, a longer chain
    """

    th = 0.5 if th is None else th
    c_file = c_id if c_file is None else c_file
    existing_classifier_id = [c.id for c in i.get_message().classifier]

    if c_id not in existing_classifier_id:  # Add classifier if not present previously in the chain

        # Get the last model in the chain
        last_id = i.get_message().classifier[-1].id  # Last classifier added to the chain
        individual_id = str(i.get_sysid())

        # Create dataset for new trigger
        data = __make_data_and_dataset(i, c_id, c_file, th)
        i.add_data(data)

        # Define trigger
        trigger_id = "%s_%s_%s" % ('trigger', individual_id, last_id)
        trigger = make.make_trigger(trigger_id, make.make_empty_classifier(data_id=data.id), component_ids=[c_id], model="probability")
        i.add_trigger(trigger)

        # Create new mutated classifier
        classifier = make.make_classifier(c_id, c_file)
        i.add_classifier(classifier)

        # Update last classifier to connect to trigger
        i.get(last_id).component_id = trigger_id


def replace_classifier(i, c_id, c_id_new, c_file = None):
    """
    Replaces a classifier in the chain for a new classifier
    :param i: Ensemble of models (system) representing an individual
    :param c_id: Classifier to be replaced
    :param c_id_new: Classifier that replaces
    :return: A mutated individual, different classifier in the chain
    """
    c_file = c_id if c_file is None else c_file
    existing_classifier_id = [c.id for c in i.get_message().classifier]

    if c_id not in existing_classifier_id:
        classifier = make.make_classifier(c_id_new, c_file)
        if i.get(c_id).HasField("component_id"):

            # Check it is a trigger
            trigger_id = i.get(c_id).component_id
            assert i.get(trigger_id).DESCRIPTOR.name == "Trigger", \
                "ERROR: Classifier in chain should be connected to trigger"

            # Replace previous data component on the system for the new one
            trigger_classifier_old = i.get(trigger_id).classifier
            old_data_id = trigger_classifier_old.data_id
            old_data_source = i.get(old_data_id).source
            th = io.read_pickle(old_data_source.val_path)['th']  # Maintain the threshold of the replaced classifier
            data = __make_data_and_dataset(i, c_id_new, c_file, th)
            __delete_dataset_and_classifier(old_data_source, trigger_classifier_old) # Removes data from previous dataset
            i.replace(old_data_id, data)

            # Empty trigger for training again during evaluation
            i.get(trigger_id).classifier.CopyFrom(make.make_empty_classifier(data_id=data.id))

            # Connect replacing classifier with trigger
            classifier.component_id = trigger_id

        i.replace(c_id, classifier)
        # Tenir en compte l'start
        if i.get_start() == c_id:
            i.set_start(c_id_new)

        # Get trigger connected to the old classifier
        trigger_names = i.get_message().trigger  # All triggers from the ensemble
        for trigger in trigger_names:
            trigger = i.get(trigger.id)
            for ic, c in enumerate(trigger.component_ids):
                if c_id == c:
                    trigger.component_ids[ic] = c_id_new


def update_threshold(i, c_id, step):
    """
    Updates the threshold of a classifier by adding step (+ or -)
    :param i: Ensemble of models (system) representing an individual
    :param c_id: Classifier affected
    :param step: Value to which increment or decrement the threshold
    :return: A mutated individual, different threshold
    """

    c_file = i.get(c_id).classifier_file
    # Update dataset and set trigger's classifier to empty
    if i.get(c_id).HasField("component_id"):
        trigger_id = i.get(c_id).component_id
        trigger = i.get(trigger_id)
        assert trigger.DESCRIPTOR.name == "Trigger", "Classifiers should be connected to triggers in the chain"
        data_source = i.get(trigger.classifier.data_id).source
        th = io.read_pickle(data_source.val_path)['th']
        new_th = th+step
        __update_dataset(c_file, data_source.train_path, data_source.test_path, data_source.val_path, new_th)
        trigger.classifier.CopyFrom(make.make_empty_classifier(data_id=trigger.classifier.data_id))  # Protobuf syntx


#   ---- VERSION 2: Extend operations for merger ----    #


