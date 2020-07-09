import Source.io_util as io
import Source.make_util as make
import Source.system_builder_serializable as sb
import Source.system_evaluator as se
import Examples.metadata_manager_results as results_manager
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


if __name__ == "__main__":

    id_fastest = "__trigger_classifier_0.10000000000000003_V001_VGG13_ref_0____trigger_classifier_0" \
                 ".4_V001_ResNeXt29_32x4d_ref_0__V001_VGG11_ref_0"

    experiment_dir = os.path.join(os.environ['FCM'], 'Examples', 'compute', 'chain_genetic_algorithm')
    metadata_file = os.path.join(experiment_dir, 'results', 'metadata.json')

    id = "7062152700584889"
    dataset = results_manager.get_fieldval_by_id(metadata_file, id, 'dataset')
    results_chain_path = os.path.join(experiment_dir, results_manager.get_results_by_id(metadata_file, id))
    results_chain = io.read_pickle(results_chain_path)
    fastest = results_chain[id_fastest]
    acc_result = fastest.test['system'].accuracy
    time_result = fastest.test['system'].time

    # Build same ensemble
    classifiers = [os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers', 'sota_models_stl10-32-dev_validation', 'V001_VGG13_ref_0'),
                   os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers', 'sota_models_stl10-32-dev_validation', 'V001_ResNeXt29_32x4d_ref_0'),
                   os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers', 'sota_models_stl10-32-dev_validation', 'V001_VGG11_ref_0')
                   ]
    classifiers_id = ['V001_VGG13_ref_0',
                      'V001_ResNeXt29_32x4d_ref_0',
                      'V001_VGG11_ref_0'
                      ]
    thresholds = [0.1, 0.4]
    trigger_ids = ['trigger_classifier_0.1_V001_VGG13_ref_0',
                   'trigger_classifier_0.4_V001_ResNeXt29_32x4d_ref_0']

    chain = build_chain(classifiers, classifiers_id, thresholds, trigger_ids, 'test_ga_chain')
    r = se.evaluate(chain, chain.get_start())
    assert acc_result == r.test['system'].accuracy, "ERROR: Accuracy of GA solution should be the same as the manual"
    assert time_result == r.test['system'].time, "ERROR: Time of GA solution should be the same as the manual"

    print("Genetic Algorithm")
    print(acc_result)
    print(time_result)
    print("Built")
    print(r.test['system'].accuracy)
    print(r.test['system'].time)