from extend_chain_operation import chain
from Source.genetic_algorithm.operations_mutation import set_threshold
from Source.system_evaluator import evaluate
from Source.system_evaluator_utils import pretty_print
from Examples.compute.chain_genetic_algorithm.utils import generate_system_id

print(chain.get_message())
print(chain.get_sysid())
set_threshold(chain, "VGG13", 0.5)
chain.set_sysid(generate_system_id(chain))
print(chain.get_sysid())
R = evaluate(chain, chain.get_start())
pretty_print(R)
set_threshold(chain, "ResNeXt29_32x4d", 0.3)
chain.set_sysid(generate_system_id(chain))
print(chain.get_sysid())
R = evaluate(chain, chain.get_start())
pretty_print(R)
set_threshold(chain, "VGG11", 0.2)
chain.set_sysid(generate_system_id(chain))
print(chain.get_sysid())
R = evaluate(chain, chain.get_start())
pretty_print(R)

