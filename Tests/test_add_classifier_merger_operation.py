import unittest
import Source.system_builder_serializable as sb
import Source.make_util as mutils
import Source.io_util as io
from Source.system_evaluator import evaluate
from Source.system_evaluator_utils import pretty_print
from Source.genetic_algorithm.operations_mutation import add_classifier_to_merger
import Source.FastComposedModels_pb2 as fcm

import unittest
import os
import numpy as np
import math


def create_ensemble():
    ensemble = sb.SystemBuilder()
    c_id = 'ResNet18'
    c_file = os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers', 'sota_models_cifar10-32-dev_validation',
                          'V001_ResNet18_ref_0.pkl')
    classifier = mutils.make_classifier(c_id, c_file)
    merger_id = 'Merger'
    merger = mutils.make_merger(merger_id, [c_id], merge_type=fcm.Merger.AVERAGE)
    ensemble.add_classifier(classifier)
    ensemble.add_merger(merger)
    ensemble.set_start('Merger')
    return ensemble


class ExtendMerger(unittest.TestCase):

    def test_accuracy(self):
        ensemble = create_ensemble()

        c_file = os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers', 'sota_models_cifar10-32-dev_validation',
                              'V001_DenseNet161_ref_0.pkl')
        c_file2 = os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers', 'sota_models_cifar10-32-dev_validation',
                     'V001_ResNet18_ref_0.pkl')

        # System Evaluator computes accuracy
        add_classifier_to_merger(ensemble, 'Merger', 'DenseNet161', c_file)
        R = evaluate(ensemble, ensemble.get_start())
        acc = R.test['system'].accuracy

        # Compute accuracy manually
        c_dict_0 = io.read_pickle(c_file2)
        c_dict_1 = io.read_pickle(c_file)

        gt = c_dict_0['test']['gt']
        logits_0 = c_dict_0['test']['logits']
        logits_1 = c_dict_1['test']['logits']

        average = (logits_0 + logits_1) / 2
        acc_manual = np.sum(np.argmax(average, 1) == gt) / len(gt)

        self.assertEqual(acc, acc_manual)

    def test_time(self):
        ensemble = create_ensemble()

        c_file = os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers', 'sota_models_cifar10-32-dev_validation',
                              'V001_DenseNet161_ref_0.pkl')
        c_file2 = os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers', 'sota_models_cifar10-32-dev_validation',
                               'V001_ResNet18_ref_0.pkl')

        # Evaluation time of ensemble
        add_classifier_to_merger(ensemble, 'Merger', 'DenseNet161', c_file)
        R = evaluate(ensemble, ensemble.get_start())
        time = R.test['system'].time

        # Compute CIFAR-10 evaluation time manually
        c_dict_0 = io.read_pickle(c_file2)
        c_dict_1 = io.read_pickle(c_file)

        time_0 = c_dict_0['metrics']['time']/128 * 5e3
        time_1 = c_dict_1['metrics']['time']/128 * 5e3

        self.assertAlmostEqual(time_1+time_0, time)

    def test_params(self):
        ensemble = create_ensemble()

        c_file = os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers', 'sota_models_cifar10-32-dev_validation',
                              'V001_DenseNet161_ref_0.pkl')
        c_file2 = os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers', 'sota_models_cifar10-32-dev_validation',
                               'V001_ResNet18_ref_0.pkl')

        # Evaluation time of ensemble
        add_classifier_to_merger(ensemble, 'Merger', 'DenseNet161', c_file)
        R = evaluate(ensemble, ensemble.get_start())
        params = R.test['system'].params

        # Compute CIFAR-10 evaluation time manually
        c_dict_0 = io.read_pickle(c_file2)
        c_dict_1 = io.read_pickle(c_file)

        params_0 = c_dict_0['metrics']['params']
        params_1 = c_dict_1['metrics']['params']

        self.assertEqual(params_0+params_1, params)

    def test_structure(self):
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
