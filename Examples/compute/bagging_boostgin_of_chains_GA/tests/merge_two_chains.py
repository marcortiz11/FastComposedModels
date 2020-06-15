import extend_chain_operation
from create_dummy_chain import build_chain
import os

chain1 = extend_chain_operation.chain

classifiers = [os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers', 'sota_models_cifar10-32-dev_validation',
                            'V001_LeNet_ref_0'),
               os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers', 'sota_models_cifar10-32-dev_validation',
                            'V001_ResNet50_ref_0')
               ]
classifiers_id = ['LeNet',
                  'ResNet50',
                  ]
thresholds = [0.6]

trigger_ids = ['trigger_classifier_0.6_LeNet']

chain2 = build_chain(classifiers, classifiers_id, thresholds, trigger_ids, 'chain2_extend_operation')

from Source.genetic_algorithm.operations_breed import merge_two_chains

merged = merge_two_chains(chain1, chain2)
merged.set_sysid(extend_chain_operation.generate_system_id(merged))

from Source.system_evaluator import evaluate
from Source.system_evaluator_utils import pretty_print
R = evaluate(merged, merged.get_start())
pretty_print(R)