import Source.make_util as make
import Source.system_builder as sb
import Source.system_evaluator as eval
import Source.io_util as io
import Source.genetic_algorithm.operations_breed as ob
import os

L = []


def update_dataset(c_file, train_path, test_path, val_path, th):
    # Create dataset
    model = io.read_pickle(c_file)
    # Test
    dataset_test = model['test']
    if th >= 0: dataset_test['th'] = th
    io.save_pickle(test_path, dataset_test)

    """
    # Train
     dataset_train = model['train']
    if th >= 0: dataset_train['th'] = th
    io.save_pickle(train_path, dataset_train)
    # Validation
    dataset_val = model['val']
    if th >= 0: dataset_val['th'] = th
    io.save_pickle(val_path, dataset_val)
    """


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
        test_path = os.path.join(data_path, id_triggers[i] + "_test.pkl")
        source = make.make_source(test_path, test_path, 2)
        data = make.make_data("data_"+id_triggers[i], 1, 1, source=source)
        update_dataset(classifiers[i], test_path, test_path, test_path, thresholds[i])
        sys.add_data(data)

        # Build trigger attached to classifier
        trigger = make.make_trigger(id_triggers[i], make.make_empty_classifier(data_id="data_"+id_triggers[i]),
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


if __name__ == '__main__':

    c0_file = os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers', 'front45_models_validation',
                           'V001_DenseNet_s3_71')
    c1_file = os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers', 'front45_models_validation',
                           'V001_DenseNet_s1_3')
    c2_file = os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers', 'front45_models_validation',
                           'V001_DenseNet_s2_32')
    data_path = os.path.join(os.environ['FCM'], 'Data')

    # ENSEMBLE SKELETON
    chain0 = build_chain([c0_file, c1_file, c2_file],
                      ['V001_DenseNet_s3_71', 'V001_DenseNet_s1_3', 'V001_DenseNet_s2_32'],
                      [1.1, 1.1],
                      ['trigger_classifier_1.1_V001_DenseNet_s3_71', 'trigger_classifier_1.1_V001_DenseNet_s1_3'],
                      'test_crossover_chain0')

    chain1 = build_chain([c0_file, c2_file],
                      ['V001_DenseNet_s3_71', 'V001_DenseNet_s2_32'],
                      [0.8],
                      ['trigger_classifier_0.8_V001_DenseNet_s3_71'],
                      'test_crossover_chain1')

    R_chain0 = eval.evaluate(chain0, chain0.get_start())
    R_chain1 = eval.evaluate(chain1, chain1.get_start())

    c = ob.singlepoint_crossover(chain0, chain1, 'V001_DenseNet_s1_3', 'V001_DenseNet_s3_71')

    from Examples.compute.chain_genetic_algorithm.utils import generate_system_id

    print(generate_system_id(c[0]))
    print("---------------------------")
    print(generate_system_id(c[1]))

    R = eval.evaluate(c[0], c[0].get_start())
    R_ = eval.evaluate(c[1], c[1].get_start())

    # Test that the offspring result is consistent
    assert R.test['system'].accuracy == R_chain1.test['system'].accuracy, "ERROR: Test error"
    assert R_.test['system'].accuracy == R_chain0.test['system'].accuracy, "ERROR: Test error"




