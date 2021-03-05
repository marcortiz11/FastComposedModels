import Source.protobuf.system_builder_serializable as sb
import Source.protobuf.make_util as mutils
import Source.io_util as io
from Source.system_evaluator import evaluate
from Source.system_evaluator_utils import  pretty_print
from Source.genetic_algorithm.operations_mutation import extend_merged_chain
from Source.genetic_algorithm.operations_mutation import replace_classifier_merger

import unittest
import os
import numpy as np
import math


def create_chain():
    ensemble = sb.SystemBuilder()
    c_id = 'ResNet18'
    c_file = os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers', 'sota_models_cifar10-32-dev_validation',
                          'V001_ResNet18_ref_0.pkl')
    classifier = mutils.make_classifier(c_id, c_file)
    ensemble.add_classifier(classifier)
    ensemble.set_start(c_id)
    return ensemble


def extend_chain(chain):
    c_id_extend = 'DenseNet161'
    c_file_extend = os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers',
                                 'sota_models_cifar10-32-dev_validation', 'V001_DenseNet161_ref_0.pkl')
    extend_merged_chain(chain, 'ResNet18', c_id_extend, 0.0, c_file_extend)


def replace(chain):
    c_file_replace = os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers',
                                  'sota_models_cifar10-32-dev_validation', 'V001_DenseNet169_ref_0.pkl')
    replace_classifier_merger(chain, 'ResNet18', 'DenseNet169', c_file=c_file_replace)


class ReplaceClassifier(unittest.TestCase):

    def test_accuracy(self):
        chain = create_chain()
        extend_chain(chain)
        replace(chain)
        R = evaluate(chain, chain.get_start())

        # Accuracy classifier by hand
        dict_classifier = io.read_pickle(
            os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers', 'sota_models_cifar10-32-dev_validation',
                         'V001_DenseNet169_ref_0.pkl'))
        acc_net0 = np.sum(np.argmax(dict_classifier['test']['logits'], 1) == dict_classifier['test']['gt']) / len(
            dict_classifier['test']['gt'])

        self.assertEqual(acc_net0, R.test['system'].accuracy)

    def test_structure_connections_chain(self):
        chain = create_chain()
        extend_chain(chain)
        replace(chain)

        correct = chain.get(chain.get_start()).component_id == 'trigger_classifier_0.0_DenseNet169' and \
                chain.get('trigger_classifier_0.0_DenseNet169').component_ids[0] == 'DenseNet161' and \
                len(chain.get('trigger_classifier_0.0_DenseNet169').component_ids) == 1
        self.assertTrue(correct)

    def test_structure_check_start(self):
        chain = create_chain()
        extend_chain(chain)
        replace(chain)

        self.assertTrue(chain.get_start() == 'DenseNet169')


if __name__ == '__main__':
    unittest.main()
