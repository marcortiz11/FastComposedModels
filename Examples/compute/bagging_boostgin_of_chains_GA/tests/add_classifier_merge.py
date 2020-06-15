import merge_two_chains
import os

merger = merge_two_chains.merged

from Source.genetic_algorithm.operations_mutation import add_classifier_to_merger
add_classifier_to_merger(merger, 'Merger', '3_PNASNetA', os.path.join(os.environ['FCM'], 'Definitions', 'Classifiers',
                                                                      'sota_models_cifar10-32-dev_validation',
                                                                      'V001_PNASNetA_ref_0'))

from Examples.compute.chain_genetic_algorithm.utils import generate_system_id
merger.set_sysid(generate_system_id(merger))
print(merger.get_sysid())