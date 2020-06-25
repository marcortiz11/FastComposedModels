from add_classifier_merge import merger
from Examples.compute.chain_genetic_algorithm.utils import generate_system_id
from Source.genetic_algorithm.operations_mutation import update_threshold

update_threshold(merger, "1_ResNet34", -0.8)
merger.set_sysid(generate_system_id(merger))
print(merger.get_sysid())

from Source.system_evaluator import evaluate
from Source.system_evaluator_utils import pretty_print
R = evaluate(merger, merger.get_start())
pretty_print(R)