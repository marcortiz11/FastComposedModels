import Source.io_util as io
import Source.system_evaluator as eval
import Source.system_evaluator_utils as eval_utils
import numpy as np


def evaluate(sys, results, c, check_classifiers, classifier_dict, input_ids, phase):
    contribution = {}

    c_dict = io.read_pickle(c.classifier_file)
    L, gt, ids = eval_utils.get_Lgtid(c_dict, phase, input_ids)
    input_ids_ = input_ids if input_ids is not None else c_dict[phase]['id']
    n_inputs = len(input_ids_)

    predictions = np.argmax(L, axis=1)
    if c.id not in results:
        results[c.id] = eval.create_metrics_classifier(c_dict, predictions, gt, n_inputs)
    else:
        eval.update_metrics_classifier(results[c.id], c_dict, predictions, gt, n_inputs)
    eval.update_metrics_system(results, c_dict, n_inputs, input_ids=input_ids, phase=phase)

    contribution['predictions'] = dict(zip(ids, predictions))
    contribution['gt'] = dict(zip(ids, gt))
    contribution['logits'] = dict(zip(ids, L))
    contribution['model'] = {}  # dict(zip(input_ids_, [[c_dict['name']]] * n_inputs))


    # Time per instance:
    input_ids_aux = input_ids if input_ids is not None else ids
    time_instance = c_dict['metrics']['time'] / 128.0
    contribution['time_instance'] = dict(
        zip(input_ids if input_ids is not None else ids, np.ones(n_inputs) * time_instance))
    if "time_instance" in c_dict[phase]:
        indices = np.where(np.isin(c_dict[phase]['id'], input_ids_aux))
        indices = indices[0] if len(indices) > 0 else []
        contribution['time_instance'].update(dict(zip(c_dict[phase]['id'][indices],
                                                      c_dict[phase]['time_instance'][indices])))

    # Now classifier can call other components
    if c.component_id != '':
        contribution_component = eval.__evaluate(sys, results, sys.get(c.component_id), check_classifiers,
                                            classifier_dict, input_ids=input_ids, phase=phase)
        contribution['gt'].update(contribution_component['gt'])
        contribution['predictions'].update(contribution_component['predictions'])
        contribution['model'].update(
            dict([(k, contribution['model'][k] + v) for k, v in contribution_component['model'].items()]))
        if classifier_dict is not None:
            contribution['logits'].update(contribution_component['logits'])
            for k, v in contribution_component['time_instance'].items():
                contribution['time_instance'][k] = contribution['time_instance'][k] + v if k in contribution[
                    'time_instance'] else v

    return contribution