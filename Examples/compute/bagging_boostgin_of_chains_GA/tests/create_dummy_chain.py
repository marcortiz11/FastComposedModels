import Source.io_util as io
import Source.make_util as make
import Source.system_builder_serializable as sb
import os


def update_dataset(c_file, train_path, test_path, val_path, th):
    # Create dataset
    model = io.read_pickle(c_file)

    # Test
    dataset_test = model['test']
    if th >= 0: dataset_test['th'] = th
    io.save_pickle(test_path, dataset_test)

    dataset_train = model['train']
    if th >= 0: dataset_train['th'] = th
    io.save_pickle(train_path, dataset_train)

    # Validation
    dataset_val = model['val']
    if th >= 0: dataset_val['th'] = th
    io.save_pickle(val_path, dataset_val)


def build_chain(classifiers, id_classifiers, thresholds, id_triggers, data_id):

    assert len(classifiers) == len(id_triggers)+1, "ERROR: Number of triggers in the chain is not consistent"
    assert len(id_triggers) == len(thresholds), "ERROR: Each trigger should be assigned a threshold"
    assert len(classifiers) == len(id_classifiers), "ERROR: Each classifier file should be assigned a classifier id"

    data_path = os.path.join(os.environ['FCM'], 'Data', data_id)

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    sys = sb.SystemBuilder(verbose=False)
    for i in range(len(classifiers)-1):

        # Create data for the trigger
        train_path = os.path.join(data_path, id_triggers[i] + "_test.pkl")
        test_path = os.path.join(data_path, id_triggers[i] + "_train.pkl")
        val_path = os.path.join(data_path, id_triggers[i] + "_val.pkl")
        source = make.make_source(train_path, test_path, 2, val_path=val_path)
        data = make.make_data("data_"+id_triggers[i], 1, 1, source=source)
        update_dataset(classifiers[i], train_path, test_path, val_path, thresholds[i])
        sys.add_data(data)

        # Build trigger attached to classifier
        trigger = make.make_trigger(id_triggers[i], make.make_empty_classifier(id='', data_id="data_"+id_triggers[i]),
                                    [id_classifiers[i+1]], model="probability")
        sys.add_trigger(trigger)

        # Build classifier
        c_file = classifiers[i]
        classifier = make.make_classifier(id_classifiers[i], c_file, component_id=id_triggers[i])
        sys.add_classifier(classifier)

        if i == 0:
            sys.set_start(id_classifiers[i])

    classifier = make.make_classifier(id_classifiers[-1], classifiers[-1])
    sys.add_classifier(classifier)
    return sys