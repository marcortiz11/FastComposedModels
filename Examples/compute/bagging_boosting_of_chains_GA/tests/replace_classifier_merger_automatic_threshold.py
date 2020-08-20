from extend_chain_operation import chain
from Source.genetic_algorithm.operations_mutation import replace_classifier_merger_automatic_threshold
from Source.system_evaluator import evaluate
from Source.system_evaluator_utils import pretty_print
from Examples.compute.chain_genetic_algorithm.utils import generate_system_id
import os

print(chain.get_message())
print(chain.get_sysid())
c_file = os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers', 'sota_models_cifar10-32-dev_validation',
                            'V001_LeNet_ref_0')
replace_classifier_merger_automatic_threshold(chain, "VGG13", "LeNet", [0.9, 0.1, 0], c_file=c_file)
chain.set_sysid(generate_system_id(chain))
print(chain.get_sysid())
R = evaluate(chain, chain.get_start())
pretty_print(R)


