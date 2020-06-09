import unittest
import Source.make_util as make
import Source.system_builder as sb
import Source.system_evaluator as eval
import Source.io_util as io
import os

def update_dataset(c_file, train_path, test_path, val_path, th):
    # Create dataset
    model = io.read_pickle(c_file)
    # Test
    dataset_test = model['test']
    if th >= 0: dataset_test['th'] = th
    io.save_pickle(test_path, dataset_test)
    # Train
    dataset_train = model['train']
    if th >= 0: dataset_train['th'] = th
    io.save_pickle(train_path, dataset_train)
    # Validation
    dataset_val = model['val']
    if th >= 0: dataset_val['th'] = th
    io.save_pickle(val_path, dataset_val)


if __name__ == '__main__':

    c0_file = os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers', 'front45_models_validation',
                           'V001_DenseNet_s3_71')
    c1_file = os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers', 'front45_models_validation',
                           'V001_DenseNet_s1_3')
    c2_file = os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers', 'front45_models_validation',
                           'V001_DenseNet_s2_32.pkl')
    data_path = os.path.join(os.environ['FCM'], 'Data')

    # ENSEMBLE SKELETON
    sys = sb.SystemBuilder(verbose=False, id="1")
    trigger0_train_dataset = os.path.join(data_path, "train_trigger0")
    trigger0_test_dataset = os.path.join(data_path, "test_trigger0")
    trigger0_val_dataset = os.path.join(data_path, "val_trigger0")

    # Classifier 0
    c0 = make.make_classifier("c0", c0_file, component_id="trigger0")
    sys.add_classifier(c0)

    # Data for trigger 0
    source = make.make_source(trigger0_train_dataset, trigger0_test_dataset, 2, val_path=trigger0_val_dataset)
    data0 = make.make_data("trigger0_data", int(5e4), int(1e4), source=source)
    update_dataset(c0_file, trigger0_train_dataset, trigger0_test_dataset, trigger0_val_dataset, 0)
    sys.add_data(data0)

    # Trigger 0
    trigger0 = make.make_trigger("trigger0", make.make_empty_classifier(data_id="trigger0_data"),
                                 ["c1"], model="probability")
    sys.add_trigger(trigger0)
    # Classifier 1
    c1 = make.make_classifier("c1", c1_file)
    sys.add_classifier(c1)

    sys.set_start("c0")
    results = eval.evaluate(sys, sys.get_start(), phases=["test", "val"])
    # eval.eval_utils.pretty_print(results)

    #----------------------------------------------

    # Test 1:  update in threshold
    import Source.genetic_algorithm.operations_mutation as om
    sys_ = sys.copy()
    sys_.set_sysid("2")
    om.update_threshold(sys_, "c0", 1)  # Now all instances go to c1, acc = acc(c1)
    results = eval.evaluate(sys_, sys_.get_start(), phases=["test", "val"])
    # eval.eval_utils.pretty_print(results)

    # Test 2: update in threshold (negative)
    sys__ = sys_.copy()
    sys__.set_sysid("3")
    om.update_threshold(sys__, "c0", -0.2)  # Now all instances go to c1, acc = acc(c1)
    results = eval.evaluate(sys__, sys__.get_start(), phases=["test", "val"])
    # eval.eval_utils.pretty_print(results)


    # Test 3: Replace component in the chain
    sys_3 = sys_.copy()
    om.replace_classifier(sys_3, "c1", "c2", c2_file)
    results = eval.evaluate(sys_3, sys_3.get_start(), phases=["test", "val"])
    eval.eval_utils.pretty_print(results)

    # Test 4: Extend the chain
    sys_4 = sys_3.copy()
    om.extend_chain_pt(sys_4, "c1", 0.8, c1_file)
    results = eval.evaluate(sys_4, sys_4.get_start(), phases=["test", "val"])
    eval.eval_utils.pretty_print(results)

    P = [sys, sys_, sys__, sys_3, sys_4]

    # Check fitting function works
    import Source.genetic_algorithm.fitting_functions as fit_func
    fit = fit_func.f1_time_penalization(P, b=2, time_constraint=4)
    print(fit)

    # Check selection works
    import Source.genetic_algorithm.selection as selection
    next_indices = selection.rank_selection(fit, 2)
    print(next_indices)

    # Replace component in the chain




