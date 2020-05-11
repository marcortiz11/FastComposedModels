import source.make_util as make
import source.system_evaluator as eval
import source.system_builder_serializable as sb
import source.io_util as io
import os


def update_dataset(model_file, ths, train_path, test_path):
    model = io.read_pickle(model_file)
    # Test
    dataset_test = model['test']
    dataset_test['ths'] = ths
    io.save_pickle(test_path, dataset_test)
    # Train
    dataset_train = model['train']
    dataset_train['ths'] = ths
    io.save_pickle(train_path, dataset_train)


if __name__ == "__main__":

    data_path = "../data/"

    # Create skeleton
    sys = sb.SystemBuilder(verbose=False)
    trigger0_train_dataset = os.path.join(data_path, "train_trigger0")
    trigger0_test_dataset = os.path.join(data_path, "test_trigger0")
    trigger1_train_dataset = os.path.join(data_path, "train_trigger1")
    trigger1_test_dataset = os.path.join(data_path, "test_trigger1")
    c0_file = "../definitions/Classifiers/sota_models_gtsrb-32-dev_validation/V001_VGG13_ref_0.pkl"
    c1_file = "../definitions/Classifiers/sota_models_gtsrb-32-dev_validation/V001_ResNet34_ref_0.pkl"
    c2_file = "../definitions/Classifiers/sota_models_gtsrb-32-dev_validation/V001_DenseNet121_ref_0.pkl"

    # Classifier/Trigger 0
    c0 = make.make_classifier("c0", c0_file, component_id="trigger0")
    sys.add_classifier(c0)
    source = make.make_source(trigger0_train_dataset, trigger0_test_dataset, 2)
    data0 = make.make_data("trigger0_data", int(5e4), int(1e4), source=source)
    sys.add_data(data0)
    trigger0 = make.make_trigger("trigger0", make.make_empty_classifier(data_id="trigger0_data"),
                                 ["c1", "c2"], model="probability_multiple_classifiers")
    sys.add_trigger(trigger0)

    # Classifier/Trigger 1
    source = make.make_source(trigger1_train_dataset, trigger1_test_dataset, 2)
    data1 = make.make_data("trigger1_data", int(5e4), int(1e4), source=source)
    sys.add_data(data1)
    update_dataset(c1_file, [0], trigger1_train_dataset, trigger1_test_dataset)
    trigger1 = make.make_trigger("trigger1", make.make_empty_classifier(data_id="trigger1_data"),
                                 ["c2"], model="probability_multiple_classifiers")
    c1 = make.make_classifier("c1", c1_file, component_id="trigger1")
    sys.add_classifier(c1)
    sys.add_trigger(trigger1)

    # Classifier 2
    c2 = make.make_classifier("c2", c2_file)
    sys.add_classifier(c2)

    # ---- TEST ---- #

    # Test 1 -> All instances executed by the bigger network
    update_dataset(c0_file, [0, 1.1], trigger0_train_dataset, trigger0_test_dataset)
    trigger0 = make.make_trigger("trigger0", make.make_empty_classifier(id="", data_id="trigger0_data"),
                                 ["c1", "c2"], model="probability_multiple_classifiers")
    sys.replace("trigger0", trigger0)
    R1 = eval.evaluate(sys, "c0")
    assert R1.test['system'].accuracy == R1.test['c2'].accuracy and \
            R1.test['system'].time == R1.test['c0'].time + R1.test['c2'].time, "Error in test 1"
    print("TEST 1: PASS")

    # Test 2 -> All instances executed by the medium network
    update_dataset(c0_file, [1.1, 0], trigger0_train_dataset, trigger0_test_dataset)
    trigger0 = make.make_trigger("trigger0", make.make_empty_classifier(id="", data_id="trigger0_data"),
                                 ["c1", "c2"], model="probability_multiple_classifiers")
    sys.replace("trigger0", trigger0)
    R2 = eval.evaluate(sys, "c0")
    assert R2.test['system'].accuracy == R2.test['c1'].accuracy and \
            R2.test['system'].time == R2.test['c0'].time + R2.test['c1'].time, "Error in test 2"
    print("TEST 2: PASS")

    # Test 3 -> All instances executed by the bigger network
    update_dataset(c1_file, [1.1], trigger1_train_dataset, trigger1_test_dataset)
    # Trigger 1
    trigger1 = make.make_trigger("trigger1", make.make_empty_classifier(data_id="trigger1_data"),
                                 ["c2"], model="probability_multiple_classifiers")
    sys.replace("trigger1", trigger1)
    R3 = eval.evaluate(sys, "c0")
    assert R3.test['system'].accuracy == R3.test['c2'].accuracy and\
            R3.test['system'].time == R3.test['c0'].time + R3.test['c2'].time + R3.test['c1'].time, "Error in test 3"
    print("TEST 3: PASS")


    # Test 4 -> All instances executed by the big network, different path in the graph
    update_dataset(c0_file, [0.6, 1.1], trigger0_train_dataset, trigger0_test_dataset)
    trigger0 = make.make_trigger("trigger0", make.make_empty_classifier(id="", data_id="trigger0_data"),
                                 ["c1", "c2"], model="probability_multiple_classifiers")
    sys.replace("trigger0", trigger0)
    R3 = eval.evaluate(sys, "c0")
    assert R3.test['system'].accuracy == R3.test['c2'].accuracy, "Error in test 4 accuracy"
    assert R3.test['system'].params == R3.test['c0'].params + R3.test['c1'].params + R3.test['c2'].params + 2, \
                                    "Error in test 4 params"
    print("TEST 4 PASS")


    # Test 5 -> Incrementing the threshold on a fully connected chain
    import numpy as np
    import examples.plot as plt

    R = {}
    for th in np.arange(0, 1.1, 0.1):
        # Send to Network 2
        update_dataset(c0_file, [th, 0], trigger0_train_dataset, trigger0_test_dataset)
        trigger0 = make.make_trigger("trigger0", make.make_empty_classifier(data_id="trigger0_data"),
                                     ["c1", "c2"], model="probability_multiple_classifiers")
        sys.replace("trigger0", trigger0)
        # Send to network 1
        update_dataset(c2_file, [th], trigger1_train_dataset, trigger1_test_dataset)
        trigger1 = make.make_trigger("trigger1", make.make_empty_classifier(data_id="trigger1_data"),
                                     ["c2"], model="probability_multiple_classifiers")
        sys.replace("trigger1", trigger1)
        R["system_"+str(th)] = eval.evaluate(sys, "c0").test

    #plt.plot_accuracy_time(R)
    #plt.show()
    print("TEST 5: PASS")

    # Test 6 -> When both are triggered
    update_dataset(c2_file, [0], trigger1_train_dataset, trigger1_test_dataset)
    trigger1 = make.make_trigger("trigger1", make.make_empty_classifier(data_id="trigger1_data"),
                                 ["c1"], model="probability_multiple_classifiers")
    sys.replace("trigger1", trigger1)
    update_dataset(c0_file, [0.6, 1.1], trigger0_train_dataset, trigger0_test_dataset)
    trigger0 = make.make_trigger("trigger0", make.make_empty_classifier(data_id="trigger0_data"),
                                 ["c2", "c1"], model="probability_multiple_classifiers")
    sys.replace("trigger0", trigger0)
    R6 = eval.evaluate(sys, "c0")

    # Check manually the accuracy result
    c0_dict = io.read_pickle(c0_file)
    L = c0_dict['test']['logits']
    dividend = np.sum(np.exp(L), axis=1)
    P = np.exp(L) / dividend[:, None]
    sort = np.sort(P, axis=1)
    diff = sort[:, -1] - sort[:, -2]
    mask = diff < 0.6
    no_mask = diff >= 0.6

    c2_dict = io.read_pickle(c2_file)
    L2 = c2_dict['test']['logits']
    pred2 = np.argmax(L2, axis=1)[mask]
    gt2 = c2_dict['test']['gt'][mask]
    correct = np.sum(pred2 == gt2)

    c1_dict = io.read_pickle(c1_file)
    L1 = c1_dict['test']['logits']
    pred1 = np.argmax(L1, axis=1)[no_mask]
    gt1 = c1_dict['test']['gt'][no_mask]
    correct += np.sum(pred1 == gt1)

    accuracy = correct/1e4
    assert accuracy == R6.test['system'].accuracy, "Error in TEST 6"
    print("TEST 6: PASS")

