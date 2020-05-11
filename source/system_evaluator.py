import source.system_evaluator_utils as eval_utils
import source.make_util as make
import source.io_util as io
import numpy as np
import warnings


class Metrics(object):
    __slots__ = ('accuracy',
                 'time',
                 'time_max',
                 'time_min',
                 'time_std',
                 'instances',  # How many instances executed by the model
                 'instance_model',  # Which model executes the instance?
                 'cm',  # Confusion matrix
                 'params',
                 'ops',)


class Results(object):
    __slots__ = ('train', 'test', 'val')


def update_metrics_classifier(m, c_dict, predictions, gt, n_inputs):
    correct = m.instances*m.accuracy
    m.instances += n_inputs
    m.accuracy = (np.sum(predictions == gt) + correct) / m.instances
    m.ops += c_dict['metrics']['ops'] * n_inputs
    m.time += c_dict['metrics']['time'] / 128 * n_inputs
    m.time_max += np.max(c_dict['metrics']['times']) / 128 * n_inputs
    m.time_min += np.min(c_dict['metrics']['times']) / 128 * n_inputs
    m.time_std += np.std(c_dict['metrics']['times']) / 128 * n_inputs


def create_metrics_classifier(c_dict, predictions, gt, n_inputs):
    m = Metrics()
    m.instances = n_inputs
    m.ops = c_dict['metrics']['ops']
    m.params = c_dict['metrics']['params']
    m.accuracy = np.sum(predictions == gt)/n_inputs if n_inputs > 0 else 0
    m.time = c_dict['metrics']['time']/128 * n_inputs
    m.time_max = np.max(c_dict['metrics']['times']) / 128 * n_inputs
    m.time_min = np.min(c_dict['metrics']['times']) / 128 * n_inputs
    m.time_std = np.std(c_dict['metrics']['times']) / 128 * n_inputs
    return m


def update_metrics_system(results, c_dict, n_inputs, phase="test", input_ids=[]):
    if "time_instance" in c_dict[phase]:
        indices = np.where(np.isin(c_dict[phase]['id'], input_ids if input_ids is not None else c_dict[phase]['id']))
        indices = indices[0] if len(indices) > 0 else []
        results['system'].time += sum(np.array(c_dict[phase]['time_instance'])[indices])
    else:
        results['system'].time += c_dict['metrics']['time'] / 128 * n_inputs
    results['system'].time_max += np.max(c_dict['metrics']['times']) / 128 * n_inputs
    results['system'].time_min += np.min(c_dict['metrics']['times']) / 128 * n_inputs
    results['system'].time_std += np.std(c_dict['metrics']['times']) / 128 * n_inputs
    results['system'].ops += c_dict['metrics']['ops']


def check_valid_classifier_structure(c_dict):
    assert "train" in c_dict.keys() and "test" in c_dict.keys(), \
        "ERROR in Classifier dictionary: Classifier should have train and test fields"
    assert "logits" in c_dict['train'].keys() and "logits" in c_dict['test'].keys() and \
            "gt" in c_dict['train'].keys() and "gt" in c_dict['test'].keys() and \
            "id" in c_dict['train'].keys() and "id" in c_dict['test'].keys(), \
            "Error in Classifier dictionary: Classifier must have logits, gt, id fields in train, test and validation"
    assert "time" in c_dict['metrics'] and \
            "ops" in c_dict['metrics'] and \
            "params" in c_dict['metrics']
    # Comprovar validation
    if "val" in c_dict.keys():
        assert "logits" in c_dict['val'].keys() and\
               "gt" in c_dict['val'].keys() and \
               "id" in c_dict['val'].keys(), \
            "Error in Classifier dictionary: Classifier must have logits, gt, id fields in train, test and validation"


def check_valid_classifier_raw_data(c_dict):
    # Train
    L_train, gt, ids = eval_utils.get_Lgtid(c_dict, "train")
    assert L_train.shape[0] == gt.shape[0] and L_train.shape[0] == len(set(ids)), \
        "ERROR in Classifier: number of instances should be consistent in train"
    assert L_train.shape[1] == len(set(gt)), \
        "ERROR in Classifier: number of classses should be consistent in train"

    # Test
    L_test, gt, ids = eval_utils.get_Lgtid(c_dict, "test")
    assert L_test.shape[0] == gt.shape[0] and L_test.shape[0] == len(set(ids)), \
        "ERROR in Classifier: number of instances should be consistent in test"
    assert L_test.shape[1] == len(set(gt)), \
        "ERROR in Classifier: number of classses should be consistent in test"

    # Validation
    if "val" in c_dict:
        L_val, gt, ids = eval_utils.get_Lgtid(c_dict, "val")
        assert L_val.shape[0] == gt.shape[0] and L_val.shape[0] == len(set(ids)), \
            "ERROR in Classifier: number of instances should be consistent in validation"
        assert L_val.shape[1] == len(set(gt)), \
            "ERROR in Classifier: number of classses should be consistent in validation"


def check_valid_classifier_metrics(c):
    classifier_dict = io.read_pickle(c.classifier_file, verbose=False)
    if 'time' not in classifier_dict['metrics']:
        warnings.warn("WARNING in Classifier: Time measurement not found, set to 0 instead")
        classifier_dict['metrics']['time'] = 0
        io.save_pickle(c.classifier_file, classifier_dict)
    if 'times' not in classifier_dict['metrics']:
        warnings.warn("WARNING in Classifier: Times list measurement not found, set to time instead")
        classifier_dict['metrics']['times'] = np.array([classifier_dict['metrics']['time']])
        io.save_pickle(c.classifier_file, classifier_dict)
    if 'params' not in classifier_dict['metrics']:
        warnings.warn("WARNING in Classifier: # Params measurement not found, set to 0 instead")
        classifier_dict['metrics']['params'] = 0
        io.save_pickle(c.classifier_file, classifier_dict)
    if 'ops' not in classifier_dict['metrics']:
        warnings.warn("WARNING in Classifier: # Ops measurement not found, set to 0 instead")
        classifier_dict['metrics']['ops'] = 0
        io.save_pickle(c.classifier_file, classifier_dict)

    classifier_dict = io.read_pickle(c.classifier_file, verbose=False)

    assert classifier_dict['metrics']['time'] >= 0, "ERROR in Classifier: Time should be positive"
    assert np.all(np.array(classifier_dict['metrics']['times']) >= 0) and len(classifier_dict['metrics']['times']) > 0, \
        "ERROR in Classifier: Time should be positive"
    assert classifier_dict['metrics']['params'] >= 0, "ERROR in Classifier: # Params should be positive"
    assert classifier_dict['metrics']['ops'] >= 0, "ERROR in Classifier: # Ops should be positive"


def check_valid_classifier(c):
    if c.HasField('classifier_file'):
        classifier_dict = io.read_pickle(c.classifier_file, verbose=False)
        check_valid_classifier_structure(classifier_dict)
        check_valid_classifier_raw_data(classifier_dict)
        check_valid_classifier_metrics(c)
    else:
        raise Exception("ERROR in Classifier: classifier_file field should be specified.")


def fill_classifier(eval, classifier_dict, contribution, contribution_train, contribution_val):

    metrics = make.make_performance_metrics(**{"time": eval.test['system'].time / len(contribution['gt'].keys()) * 128,
                                               "ops": eval.test['system'].ops / len(contribution['gt'].keys()),
                                               "params": eval.test['system'].params})
    # Test raw data
    test_raw = make.make_classifier_raw_data([], [], [])
    if contribution is not None:
        keys_test = [key for key in contribution['logits'].keys()]
        logits = [contribution['logits'][key] for key in keys_test]
        gt = [contribution['gt'][key] for key in keys_test]
        test_raw = make.make_classifier_raw_data(logits, gt, keys_test)
        classifier_dict['test']['time_instance'] = \
            np.array([contribution['time_instance'][key] for key in keys_test])

    # Train raw data
    train_raw = make.make_classifier_raw_data([], [], [])
    if contribution_train is not None:
        keys_train = [key for key in contribution_train['logits'].keys()]
        logits = [contribution_train['logits'][key] for key in keys_train]
        gt = [contribution_train['gt'][key] for key in keys_train]
        train_raw = make.make_classifier_raw_data(logits, gt, keys_train)
        classifier_dict['train']['time_instance'] = np.array(
            [contribution_train['time_instance'][key] for key in keys_train])

    # Validation raw data
    val_raw = make.make_classifier_raw_data([], [], [])
    if contribution_val is not None:
        keys_val = [key for key in contribution_val['logits'].keys()]
        logits = [contribution_val['logits'][key] for key in keys_val]
        gt = [contribution_val['gt'][key] for key in keys_val]
        val_raw = make.make_classifier_raw_data(logits, gt, keys_val)
        classifier_dict['val']['time_instance'] = np.array(
            [contribution_val['time_instance'][key] for key in keys_val])

    # Fill the dict
    classifier_dict.update(make.make_classifier_dict(classifier_dict['name'], "", train_raw, test_raw, metrics, val_data=val_raw))


def __evaluate(sys, results, c, check_classifiers, classifier_dict, input_ids=None, phase="test"):

    assert phase in ["train", "test", "val"], "Error: Phase in evaluation should be train, test or validation"

    if c.DESCRIPTOR.name == "Merger":
        import source.merger_evaluator as merger_eval
        contribution = merger_eval.evaluate(sys, results, c,
                                            check_classifiers, classifier_dict,
                                            input_ids=input_ids, phase=phase)
    elif c.DESCRIPTOR.name == "Classifier":
        import source.classifier_evaluator as classifier_eval
        if check_classifiers:
            check_valid_classifier(c)
        contribution = classifier_eval.evaluate(sys, results, c, check_classifiers, classifier_dict, input_ids, phase)

    elif c.DESCRIPTOR.name == "Trigger":
        import source.trigger_evaluator as trigger_eval
        eval_utils.train_trigger(sys, c)
        if check_classifiers:
            check_valid_classifier(c.classifier)
        contribution = trigger_eval.evaluate(sys, results, c, check_classifiers, classifier_dict, input_ids, phase)

    return contribution


def evaluate(sys, start_id, check_classifiers=False, evaluate_train=False, classifier_dict=None, phases=["test"]):

    """
    :param sys: Built system to evaluate
    :param start_id: component id on where to start the evaluation
    :param check_classifiers: Evaluate and check the consistency of the classifiers
    :param evaluate_train: Perform evaluation on train data
    :param classifier_dict: Build a classifier resulting form the system as a dictionary

    :return: Struct results for test and train with the performance
            of each component and the system in general
    """

    eval = Results()
    eval.train = dict()
    eval.test = dict()
    eval.val = dict()

    # TODO: Create a function for the following 30 lines
    eval.test['system'] = Metrics()
    eval.test['system'].time = 0
    eval.test['system'].accuracy = 0
    eval.test['system'].params = 0
    eval.test['system'].ops = 0
    eval.test['system'].time_max = 0
    eval.test['system'].time_min = 0
    eval.test['system'].time_std = 0

    eval.train['system'] = Metrics()
    eval.train['system'].time = 0
    eval.train['system'].accuracy = 0
    eval.train['system'].params = 0
    eval.train['system'].ops = 0
    eval.train['system'].time_max = 0
    eval.train['system'].time_min = 0
    eval.train['system'].time_std = 0

    eval.val['system'] = Metrics()
    eval.val['system'].time = 0
    eval.val['system'].accuracy = 0
    eval.val['system'].params = 0
    eval.val['system'].ops = 0
    eval.val['system'].time_max = 0
    eval.val['system'].time_min = 0
    eval.val['system'].time_std = 0

    component = sys.get(start_id)

    if "test" in phases or not phases:  # Si no s'especifica, sempre evalua el dataset de test
        contribution = __evaluate(sys, eval.test, component, check_classifiers, classifier_dict)
        eval.test['system'].accuracy = eval_utils.get_accuracy_dictionary(contribution['predictions'], contribution['gt'])
        eval.test['system'].params = sum([classifier_eval.params for classifier_id, classifier_eval in eval.test.items()
                                          if classifier_id != 'system'])
        # eval.test['system'].instance_model = contribution['model']

    contribution_train = None
    if "train" in phases or evaluate_train:
        contribution_train = __evaluate(sys, eval.train, component, check_classifiers, classifier_dict, phase="train")
        eval.train['system'].accuracy = eval_utils.get_accuracy_dictionary(contribution_train['predictions'], contribution_train['gt'])
        eval.train['system'].params = sum([classifier_eval.params for classifier_id, classifier_eval in eval.train.items() \
                                          if classifier_id != 'system'])
        # eval.train['system'].instance_model = contribution_train['model']

    contribution_val = None
    if "val" in phases:
        contribution_val = __evaluate(sys, eval.val, component, check_classifiers, classifier_dict, phase="val")
        eval.val['system'].accuracy = eval_utils.get_accuracy_dictionary(contribution_val['predictions'], contribution_val['gt'])
        eval.val['system'].params = sum([classifier_eval.params for classifier_id, classifier_eval in eval.val.items() \
                                            if classifier_id != 'system'])
        # eval.val['system'].instance_model = contribution_val['model']

    # TODO: Falta que validation set es pugui guardar com a diccionari
    if classifier_dict is not None:
        fill_classifier(eval, classifier_dict, contribution, contribution_train, contribution_val)

    return eval
