from create_dummy_chain import build_chain
from Source.system_evaluator import evaluate
from Source.system_evaluator_utils import pretty_print
import os


classifiers = [os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers', 'sota_models_cifar10-32-dev_validation',
                            'V001_VGG13_ref_0'),
               os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers', 'sota_models_cifar10-32-dev_validation',
                            'V001_ResNeXt29_32x4d_ref_0'),
               os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers', 'sota_models_cifar10-32-dev_validation',
                            'V001_VGG11_ref_0')
               ]
classifiers_id = ['VGG13',
                  'ResNeXt29_32x4d',
                  'VGG11'
                  ]
thresholds = [0.9, 0.8]

trigger_ids = ['trigger_classifier_0.1_VGG13',
               'trigger_classifier_0.4_ResNeXt29_32x4d']

chain = build_chain(classifiers, classifiers_id, thresholds, trigger_ids, 'chain_extend_operation')

# Test automatic id generator
from Examples.compute.chain_genetic_algorithm.utils import generate_system_id
chain_id = generate_system_id(chain)

# Test new extend chain operation
from Source.genetic_algorithm.operations_mutation import extend_merged_chain
c_id_new = 'DenseNet-161'
c_file_new = os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers', 'sota_models_cifar10-32-dev_validation', 'V001_DenseNet161_ref_0')
extend_merged_chain(chain, classifiers_id[-1], c_id_new, 0.9, c_file_new=c_file_new)
chain.set_sysid(generate_system_id(chain))

# Evaluate
R = evaluate(chain, chain.get_start())
