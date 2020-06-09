from Examples import metadata_manager_results as results_manager
from Source.genetic_algorithm.fitting_functions import f1_time_param_penalization
from Source import io_util as io
import matplotlib.pyplot as plt
import numpy as np
import os


def fitness_evolution(individuals_fitness_generation, R):
    fit = []
    keys = list(R.keys())
    time = np.array([R[key].test['system'].time for key in keys])
    max_time_r = keys[np.argmax(time)]
    params = np.array([R[key].test['system'].params for key in keys])
    max_params_r = keys[np.argmax(params)]

    for generation in individuals_fitness_generation:
        ids = generation[0]
        R_ids = [R[id] for id in ids]
        R_ids += [R[max_time_r], R[max_params_r]]
        fit_ids = f1_time_param_penalization(R_ids, a=5, phase="validation")
        fit += [(max(fit_ids), sum(fit_ids)/len(fit_ids), min(fit_ids))]
    return fit


def percentage_replacement_offspring(individuals_fitness_generation):
    percentages = [1, ]
    generation = 1
    while generation < len(individuals_fitness_generation):
        id_current_gen = set(individuals_fitness_generation[generation][0])
        id_previus_gen = set(individuals_fitness_generation[generation-1][0])
        percentages.append(1 - len(id_current_gen.intersection(id_previus_gen))/len(id_current_gen))
        generation += 1
    return percentages


def percentage_repeated_individuals(individuals_fitness_generation):
    percentages = [1, ]
    generation = 1
    while generation < len(individuals_fitness_generation):
        id_current_gen = set(individuals_fitness_generation[generation][0])
        id_previus_gen = set(individuals_fitness_generation[generation-1][0])
        percentages.append(1 - len(id_current_gen.intersection(id_previus_gen))/len(id_current_gen.union(id_previus_gen)))
        generation += 1
    return percentages


def plot_fitness_evolution(individuals_fitness_generation, R):
    X = range(len(individuals_fitness_generation))
    Y = fitness_evolution(individuals_fitness_generation, R)

    Y_max = [y[0] for y in Y]
    Y_average = [y[1] for y in Y]
    Y_min = [y[2] for y in Y]

    plt.figure()
    plt.title("Evolution of the Fitness value")
    plt.xlabel("Generations")
    plt.ylabel("Validation Fitness Value")
    plt.scatter(X, Y_max, facecolor='gray', marker="_")
    plt.scatter(X, Y_average, facecolor='blue')
    plt.scatter(X, Y_min, facecolor='gray', marker="_")
    plt.vlines(X, Y_min, Y_max, color='gray', linewidth=0.5)
    plt.show()


def plot_offspring_replacement_evolution(individuals_fitness_generation):
    X = range(len(individuals_fitness_generation))
    Y = percentage_replacement_offspring(individuals_fitness_generation)

    plt.figure()
    plt.title("Offspring survival rate")
    plt.xlabel("Generations")
    plt.ylabel("Rate")
    plt.plot(X, Y)
    plt.show()


def plot_repeated_individuals_evolution(individuals_fitness_generation):
    X = range(len(individuals_fitness_generation))
    Y = percentage_repeated_individuals(individuals_fitness_generation)

    plt.figure()
    plt.title("Ratio unique individuals")
    plt.xlabel("Generations")
    plt.ylabel("Ratio")
    plt.plot(X, Y)
    plt.show()


if __name__ == "__main__":

    experiment = 'chain_genetic_algorithm_multinode'
    id = 3176850520651660

    # 1) G.A. Chain ensembles
    GA_results_metadata_file = os.path.join(os.environ['FCM'],
                                            'Examples',
                                            'compute',
                                            experiment,
                                            'results',
                                            'metadata.json')

    # Get evaluation results from query
    GA_res_loc = results_manager.get_results_by_id(GA_results_metadata_file, id)
    individuals_fitness_generation = io.read_pickle(os.path.join(GA_res_loc, 'individuals_fitness_per_generation.pkl'))
    R = io.read_pickle(os.path.join(GA_res_loc, 'results_ensembles.pkl'))

    # Plot performance of the Genetic Algorithm through generations
    plot_fitness_evolution(individuals_fitness_generation, R)
    plot_offspring_replacement_evolution(individuals_fitness_generation)
    plot_repeated_individuals_evolution(individuals_fitness_generation)
