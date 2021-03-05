import Source.protobuf.system_builder_serializable as sb
import Source.protobuf.make_util as mutils
import Source.io_util as io
from Source.system_evaluator import evaluate
from Source.system_evaluator_utils import  pretty_print
from Source.genetic_algorithm.operations_mutation import extend_merged_chain

import unittest
import os
import numpy as np
import math


class ExtendChain(unittest.TestCase):

    def __create_ensemble(self):
        ensemble = sb.SystemBuilder()
        c_id = 'ResNet18'
        c_file = os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers', 'sota_models_cifar10-32-dev_validation',
                              'V001_ResNet18_ref_0.pkl')
        classifier = mutils.make_classifier(c_id, c_file)
        ensemble.add_classifier(classifier)
        ensemble.set_start(c_id)
        return ensemble

    def test_threshold_0(self):
        chain = self.__create_ensemble()
        c_id_extend = 'DenseNet-161'
        c_file_extend = os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers',
                                     'sota_models_cifar10-32-dev_validation', 'V001_DenseNet161_ref_0.pkl')
        extend_merged_chain(chain, 'ResNet18', c_id_extend, 0, c_file_extend)
        R = evaluate(chain, chain.get_start())
        acc_chain = R.test['system'].accuracy
        time_chain = R.test['system'].time
        ops_chain = R.test['system'].ops
        params_chain = R.test['system'].params


        # Accuracy classifier by hand
        dict_classifier = io.read_pickle(os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers', 'sota_models_cifar10-32-dev_validation',
                              'V001_ResNet18_ref_0.pkl'))
        acc_net0 = np.sum(np.argmax(dict_classifier['test']['logits'], 1) == dict_classifier['test']['gt']) / len(
            dict_classifier['test']['gt'])
        time_net0 = dict_classifier['metrics']['time']
        params_net0 = dict_classifier['metrics']['params']
        ops_net0 = dict_classifier['metrics']['ops']

        # Accuracy classifier by hand
        dict_classifier = io.read_pickle(c_file_extend)
        params_net1 = dict_classifier['metrics']['params']


        correct = acc_chain == acc_net0 and \
                    math.isclose(time_net0/128, time_chain/5e3, rel_tol=1e-09) and \
                    ops_net0 == ops_chain/5e3 and\
                    params_chain == params_net0 + params_net1 + 1

        self.assertEqual(correct, True)

    def test_threshold_1(self):
        chain = self.__create_ensemble()
        c_id_extend = 'DenseNet-161'
        c_file_extend = os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers',
                                     'sota_models_cifar10-32-dev_validation', 'V001_DenseNet161_ref_0.pkl')
        extend_merged_chain(chain, 'ResNet18', c_id_extend, 1.1, c_file_extend)
        R = evaluate(chain, chain.get_start())
        acc_chain = R.test['system'].accuracy
        time_chain = R.test['system'].time
        ops_chain = R.test['system'].ops
        params_chain = R.test['system'].params


        # Accuracy classifier by hand
        dict_classifier = io.read_pickle(os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers', 'sota_models_cifar10-32-dev_validation',
                              'V001_ResNet18_ref_0.pkl'))
        acc_net0 = np.sum(np.argmax(dict_classifier['test']['logits'], 1) == dict_classifier['test']['gt']) / len(
            dict_classifier['test']['gt'])
        time_net0 = dict_classifier['metrics']['time']
        params_net0 = dict_classifier['metrics']['params']
        ops_net0 = dict_classifier['metrics']['ops']

        # Accuracy classifier by hand
        dict_classifier = io.read_pickle(c_file_extend)
        acc_net1 = np.sum(np.argmax(dict_classifier['test']['logits'], 1) == dict_classifier['test']['gt']) / len(
            dict_classifier['test']['gt'])
        time_net1 = dict_classifier['metrics']['time']
        params_net1 = dict_classifier['metrics']['params']
        ops_net1 = dict_classifier['metrics']['ops']

        correct = acc_chain == acc_net1 and \
                    math.isclose(time_net0/128 + time_net1/128, time_chain/5e3) and \
                    ops_chain/5e3 == ops_net0 + ops_net1 and \
                    params_chain == params_net0 + params_net1 + 1

        self.assertEqual(correct, True)

    def test_structure(self):

        chain = self.__create_ensemble()
        c_id_extend = 'DenseNet-161'
        c_file_extend = os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers',
                                     'sota_models_cifar10-32-dev_validation', 'V001_DenseNet161_ref_0.pkl')
        extend_merged_chain(chain, 'ResNet18', c_id_extend, 1.1, c_file_extend)

        correct_structure = True

        # Check connections between components
        correct_structure = correct_structure and \
                            "trigger_classifier_1.1_ResNet18" == chain.get(chain.get_start()).component_id

        correct_structure = correct_structure and \
                            len(chain.get("trigger_classifier_1.1_ResNet18").component_ids) == 1

        correct_structure = correct_structure and \
                            chain.get("trigger_classifier_1.1_ResNet18").component_ids[0] == c_id_extend

        self.assertEqual(correct_structure, True)


if __name__ == '__main__':
    unittest.main()
