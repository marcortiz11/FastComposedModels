from Examples import metadata_manager_results as results_manager
from Source.genetic_algorithm.fitting_functions import f1_time_param_penalization
from Source import io_util as io
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.cm as cm

colors = cm.rainbow(np.linspace(0, 1, 10))


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
        fit_ids = f1_time_param_penalization(R_ids, a=5, phase="val")
        fit += [(max(fit_ids[:-2]), sum(fit_ids[:-2])/len(fit_ids[:-2]), min(fit_ids[:-2]))]
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
        percentages.append(len(id_current_gen)/len(individuals_fitness_generation[generation][0]))
        generation += 1
    return percentages


def get_accuracy_evolution(individuals_fitness_generation, R):
    acc = []
    for generation in individuals_fitness_generation:
        ids = generation[0]
        R_ids = [R[id] for id in ids]
        acc_ids = [r.test['system'].accuracy for r in R_ids]
        acc += [(max(acc_ids), sum(acc_ids) / len(acc_ids), min(acc_ids))]
    return acc


def get_time_evolution(individuals_fitness_generation, R):
    Y = []
    for generation in individuals_fitness_generation:
        ids = generation[0]
        R_ids = [R[id] for id in ids]
        acc_ids = [r.test['system'].time for r in R_ids]
        Y += [(max(acc_ids), sum(acc_ids) / len(acc_ids), min(acc_ids))]
    return Y


def get_params_evolution(individuals_fitness_generation, R):
    Y = []
    for generation in individuals_fitness_generation:
        ids = generation[0]
        R_ids = [R[id] for id in ids]
        acc_ids = [r.test['system'].params for r in R_ids]
        Y += [(max(acc_ids), sum(acc_ids) / len(acc_ids), min(acc_ids))]
    return Y


def plot_fitness_evolution(individuals_fitness_generation, R, id=None):
    X = range(len(individuals_fitness_generation))
    Y = fitness_evolution(individuals_fitness_generation, R)

    Y_max = [y[0] for y in Y]
    Y_average = [y[1] for y in Y]
    Y_min = [y[2] for y in Y]

    sub_plt = plt.subplot(2, 3, 1)
    sub_plt.grid(True)
    sub_plt.set_title("Evolution of the Fitness value")
    sub_plt.set_xlabel("Generations")
    sub_plt.set_ylabel("Validation Fitness Value")
    # sub_plt.scatter(X, Y_max, facecolor='gray', marker="_")
    sub_plt.scatter(X, Y_average, s=5, label=id)
    # sub_plt.scatter(X, Y_min, facecolor='gray', marker="_")
    # sub_plt.vlines(X, Y_min, Y_max, color='gray', linewidth=0.5)
    sub_plt.legend()


def plot_offspring_replacement_evolution(individuals_fitness_generation, id=None):
    X = range(len(individuals_fitness_generation))
    Y = percentage_replacement_offspring(individuals_fitness_generation)

    sub_plt = plt.subplot(2, 3, 2)
    sub_plt.grid(True)
    sub_plt.set_title("Offspring survival rate")
    sub_plt.set_xlabel("Generations")
    sub_plt.set_ylabel("Rate")
    sub_plt.set_ylim(bottom=0, top=1)
    sub_plt.plot(X, Y, label=id)
    sub_plt.legend()


def plot_repeated_individuals_evolution(individuals_fitness_generation, id=None):
    X = range(len(individuals_fitness_generation))
    Y = percentage_repeated_individuals(individuals_fitness_generation)

    sub_plt = plt.subplot(2, 3, 3)
    sub_plt.grid(True)
    sub_plt.set_title("Ratio unique individuals")
    sub_plt.set_xlabel("Generations")
    sub_plt.set_ylabel("Ratio")
    sub_plt.set_ylim(bottom=0, top=1)
    sub_plt.plot(X, Y, label=id)
    sub_plt.legend()


def plot_accuracy_evolution(individuals_fitness_generation, R, id=None):
    X = range(len(individuals_fitness_generation))
    Y = get_accuracy_evolution(individuals_fitness_generation, R)

    Y_avg = [y[1] for y in Y]

    sub_plt = plt.subplot(2, 3, 4)
    sub_plt.grid(True)
    sub_plt.set_title("Accuracy Evolution")
    sub_plt.set_xlabel("Generations")
    sub_plt.set_ylabel("Test Accuracy")
    sub_plt.set_ylim(bottom=0, top=1)
    sub_plt.plot(X, Y_avg, label=id)
    sub_plt.legend()


def plot_time_evolution(individuals_fitness_generation, R, id=None):
    X = range(len(individuals_fitness_generation))
    Y = get_time_evolution(individuals_fitness_generation, R)

    Y_avg = [y[1] for y in Y]

    sub_plt = plt.subplot(2, 3, 5)
    sub_plt.grid(True)
    sub_plt.set_title("Time Evolution")
    sub_plt.set_xlabel("Generations")
    sub_plt.set_ylabel("Expected inference time")
    sub_plt.plot(X, Y_avg, label=id)
    sub_plt.legend()


def plot_parameters_evolution(individuals_fitness_generation, R, id=None):
    X = range(len(individuals_fitness_generation))
    Y = get_params_evolution(individuals_fitness_generation, R)

    Y_avg = [y[1] for y in Y]

    sub_plt = plt.subplot(2, 3, 6)
    sub_plt.grid(True)
    sub_plt.set_title("Parameters Evolution")
    sub_plt.set_xlabel("Generations")
    sub_plt.set_ylabel("Parameters")
    sub_plt.plot(X, Y_avg, label=id)
    sub_plt.legend()


if __name__ == "__main__":

    experiment = 'genetic_algorithm_multinode'
    ids = [6528370110150120, 333080722593280, 5013575983576202]

    # 1) G.A. Chain ensembles
    GA_results_metadata_file = os.path.join(os.environ['FCM'],
                                            'Examples',
                                            'compute',
                                            experiment,
                                            'results',
                                            'metadata.json')

    plt.figure()

    # Get evaluation results from query
    for id in ids:
        GA_res_loc = results_manager.get_results_by_id(GA_results_metadata_file, id)
        individuals_fitness_generation = io.read_pickle(os.path.join(GA_res_loc, 'individuals_fitness_per_generation.pkl'))
        R = io.read_pickle(os.path.join(GA_res_loc, 'results_ensembles.pkl'))

        # Plot performance of the Genetic Algorithm through generations
        #plot_fitness_evolution(individuals_fitness_generation, R, id)
        plot_offspring_replacement_evolution(individuals_fitness_generation, id)
        plot_repeated_individuals_evolution(individuals_fitness_generation, id)
        plot_accuracy_evolution(individuals_fitness_generation, R, id)
        plot_time_evolution(individuals_fitness_generation, R, id)
        plot_parameters_evolution(individuals_fitness_generation, R, id)



    plt.show()
