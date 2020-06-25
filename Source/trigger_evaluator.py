import Source.io_util as io
import Source.system_evaluator as eval
import Source.system_evaluator_utils as eval_utils
import numpy as np


def evaluate(sys, results, c, check_classifiers, classifier_dict, input_ids, phase):
    contribution = {}

    c_dict = io.read_pickle(c.classifier.classifier_file)
    L, gt, ids = eval_utils.get_Lgtid(c_dict, phase, input_ids)
    n_inputs = len(input_ids) if input_ids is not None else len(c_dict[phase]['id'])
    predictions_trigger = np.argmax(L, axis=1)

    results[c.id] = eval.create_metrics_classifier(c_dict, predictions_trigger, gt, n_inputs)
    eval.update_metrics_system(results, c_dict, n_inputs, input_ids=input_ids, phase=phase)

    contribution['gt'] = {}
    contribution['predictions'] = {}
    contribution['logits'] = {}
    contribution['time_instance'] = {}
    contribution['model'] = {}

    for i, c_id in enumerate(c.component_ids):
        c_next = sys.get(c_id)
        mask = (predictions_trigger == i)
        ids_next = ids[mask]

        contribution_component = eval.__evaluate(sys, results, c_next, check_classifiers, classifier_dict,
                                            input_ids=ids_next, phase=phase)
        contribution['gt'].update(contribution_component['gt'])
        contribution['predictions'].update(contribution_component['predictions'])
        contribution['logits'].update(contribution_component['logits'])

        if contribution['model']:
            contribution['model'].update(
                dict([(k, contribution['model'][k] + v) for k, v in contribution_component['model'].items()]))
        else:
            contribution['model'].update(contribution_component['model'])

        if classifier_dict is not None:
            contribution['logits'].update(contribution_component['logits'])
            for k, v in contribution_component['time_instance'].items():
                contribution['time_instance'][k] = contribution['time_instance'][k] + v if k in contribution[
                    'time_instance'] else v

    return contribution

