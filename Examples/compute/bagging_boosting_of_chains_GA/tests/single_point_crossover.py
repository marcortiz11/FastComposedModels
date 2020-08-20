from Examples.compute.bagging_boosting_of_chains_GA.tests.merge_two_chains import merged, chain1
from Source.genetic_algorithm.operations_breed import singlepoint_crossover
from Examples.compute.chain_genetic_algorithm.utils import generate_system_id

print(merged.get_sysid())
print(chain1.get_sysid())


# TEST 1: Point of crossover at the head of the chain
offspring = singlepoint_crossover(merged, chain1, '1_ResNet34', 'VGG13')
print(generate_system_id(offspring[0]))
print(generate_system_id(offspring[1]))

# TEST 2: Point of crossover at the tail of the chain
offspring = singlepoint_crossover(merged, chain1, '1_VGG19', 'DenseNet-161')
print(generate_system_id(offspring[0]))
print(generate_system_id(offspring[1]))

# TEST 3: Point of crossover in between
offspring = singlepoint_crossover(merged, chain1, 'ResNeXt29_32x4d', 'VGG13')
print(generate_system_id(offspring[0]))
# print(generate_system_id(offspring[1]))
