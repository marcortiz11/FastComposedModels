from Examples import metadata_manager_results as results_manager
from Source.genetic_algorithm.fitting_functions import f1_time_param_penalization
from Source.genetic_algorithm.fitting_functions import f2_time_param_penalization
from Source import io_util as io
import matplotlib.pyplot as plt
import numpy as np
import os


def fitness_evolution(individuals_fitness_generation, R, GA_params):
    fit = []
    fit_plain = []
    keys = list(R.keys())
    time = np.array([R[key].test['system'].time for key in keys])
    max_time_r = keys[np.argmax(time)]
    min_time_r = keys[np.argmin(time)]
    params = np.array([R[key].test['system'].params for key in keys])
    max_params_r = keys[np.argmax(params)]
    min_params_r = keys[np.argmin(params)]

    for generation in individuals_fitness_generation:
        ids = generation[0]
        R_ids = [R[id] for id in ids]
        R_ids += [R[max_time_r], R[max_params_r], R[min_time_r], R[min_params_r]]
        fit_ids = f2_time_param_penalization(R_ids, GA_params['a'], phase="test")
        fit_plain.append(np.array(fit_ids[:-4]))
        fit += [(max(fit_ids[:-4]), sum(fit_ids[:-4])/len(fit_ids[:-4]), min(fit_ids[:-4]))]
    return fit, fit_plain


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
    acc_plain = []
    for generation in individuals_fitness_generation:
        ids = generation[0]
        R_ids = [R[id] for id in ids]
        acc_ids = [r.test['system'].accuracy for r in R_ids]
        acc_plain.append(np.array(acc_ids))
        acc += [(max(acc_ids), sum(acc_ids) / len(acc_ids), min(acc_ids))]
    return acc, acc_plain


def get_time_evolution(individuals_fitness_generation, R):
    Y = []
    Y_plain = []
    for generation in individuals_fitness_generation:
        ids = generation[0]
        R_ids = [R[id] for id in ids]
        time_gen = [r.test['system'].time for r in R_ids]
        Y_plain.append(np.array(time_gen))
        Y += [(max(time_gen), sum(time_gen) / len(time_gen), min(time_gen))]
    return Y, Y_plain


def get_params_evolution(individuals_fitness_generation, R):
    Y = []
    Y_plain = []
    for generation in individuals_fitness_generation:
        ids = generation[0]
        R_ids = [R[id] for id in ids]
        params_gen = [r.test['system'].params for r in R_ids]
        Y_plain.append(params_gen)
        Y += [(max(params_gen), sum(params_gen) / len(params_gen), min(params_gen))]
    return Y, Y_plain


def plot_fitness_evolution(individuals_fitness_generation, R, GA_params, id=None):
    X = range(len(individuals_fitness_generation))
    Y, _= fitness_evolution(individuals_fitness_generation, R, GA_params)

    Y_max = [y[0] for y in Y]
    Y_average = [y[1] for y in Y]
    Y_min = [y[2] for y in Y]

    sub_plt = plt.subplot(2, 3, 1)
    sub_plt.grid(True)
    sub_plt.set_xlabel("Generations")
    sub_plt.set_ylabel("Test Fitness Value")
    # sub_plt.scatter(X, Y_max, facecolor='gray', marker="_")
    sub_plt.scatter(X, Y_average, s=5, label=id)
    print(Y_average)
    # sub_plt.scatter(X, Y_min, facecolor='gray', marker="_")
    # sub_plt.vlines(X, Y_min, Y_max, color='gray', linewidth=0.5)
    # sub_plt.legend()


def plot_offspring_replacement_evolution(individuals_fitness_generation, id=None):
    X = range(len(individuals_fitness_generation))
    Y = percentage_replacement_offspring(individuals_fitness_generation)

    sub_plt = plt.subplot(2, 3, 2)
    sub_plt.grid(True)
    sub_plt.set_xlabel("Generations")
    sub_plt.set_ylabel("Offspring survival rate")
    sub_plt.set_ylim(bottom=0, top=1)
    sub_plt.plot(X, Y, label=id)
    # sub_plt.legend()


def plot_repeated_individuals_evolution(individuals_fitness_generation, id=None):
    X = range(len(individuals_fitness_generation))
    Y = percentage_repeated_individuals(individuals_fitness_generation)

    sub_plt = plt.subplot(2, 3, 3)
    sub_plt.grid(True)
    sub_plt.set_xlabel("Generations")
    sub_plt.set_ylabel("Ratio Unique")
    sub_plt.set_ylim(bottom=0, top=1)
    sub_plt.plot(X, Y, label=id)
    sub_plt.legend()


def plot_accuracy_evolution(individuals_fitness_generation, R, id=None):
    X = range(len(individuals_fitness_generation))
    Y, _ = get_accuracy_evolution(individuals_fitness_generation, R)

    Y_avg = [y[1] for y in Y]

    sub_plt = plt.subplot(2, 3, 4)
    sub_plt.grid(True)
    sub_plt.set_xlabel("Generations")
    sub_plt.set_ylabel("Test Accuracy")
    sub_plt.plot(X, Y_avg, label=id)
    # sub_plt.legend()


def plot_accuracy_fittest_evolution(individuals_fitness_generation, R, GA_params, id=None):
    X = range(len(individuals_fitness_generation))
    _, Y = get_accuracy_evolution(individuals_fitness_generation, R)
    _, fit = fitness_evolution(individuals_fitness_generation, R, GA_params)
    best = [np.argmax(fit_i) for fit_i in fit]
    Y = [y[best[i]] for i, y in enumerate(Y)]

    sub_plt = plt.subplot(2, 3, 4)
    sub_plt.grid(True)
    sub_plt.set_xlabel("Generations")
    sub_plt.set_ylabel("Test Accuracy")
    sub_plt.plot(X, Y, label=id)
    # sub_plt.plot(X, Y_max, linestyle='--')
    # sub_plt.legend()


def plot_accuracy_fittest_normalized_evolution(individuals_fitness_generation, R, GA_params, id=None):
    X = range(len(individuals_fitness_generation))
    _, Y = get_accuracy_evolution(individuals_fitness_generation, R)
    _, fit = fitness_evolution(individuals_fitness_generation, R, GA_params)
    best = [np.argmax(fit_i) for fit_i in fit]
    Y = [(y[best[i]]-0.5274)*100 for i, y in enumerate(Y)]

    sub_plt = plt.subplot(2, 3, 4)
    sub_plt.grid(True)
    sub_plt.set_xlabel("Generations")
    sub_plt.set_ylabel("Increase test Accuracy (%)")
    sub_plt.plot(X, Y, label=id)
    # sub_plt.legend()


def plot_time_evolution(individuals_fitness_generation, R, id=None):
    X = range(len(individuals_fitness_generation))
    Y = get_time_evolution(individuals_fitness_generation, R)

    Y_avg = [y[1] for y in Y]

    sub_plt = plt.subplot(2, 3, 5)
    sub_plt.grid(True)
    sub_plt.set_xlabel("Generations")
    sub_plt.set_ylabel("Expected inference time")
    sub_plt.plot(X, Y_avg, label=id)
    # sub_plt.legend()


def plot_time_fittest_normalized_evolution(individuals_fitness_generation, R, GA_params, id=None):
    X = range(len(individuals_fitness_generation))
    _, Y = get_time_evolution(individuals_fitness_generation, R)
    _, fit = fitness_evolution(individuals_fitness_generation, R, GA_params)
    best = [np.argmax(fit_i) for fit_i in fit]
    Y = [8.37/y[best[i]] for i, y in enumerate(Y)]

    sub_plt = plt.subplot(2, 3, 5)
    sub_plt.grid(True)
    sub_plt.set_xlabel("Generations")
    sub_plt.set_ylabel("Speedup")
    sub_plt.plot(X, Y, label=id)
    # sub_plt.plot(X, Y_max, linestyle='--')
    # sub_plt.legend()


def plot_parameters_evolution(individuals_fitness_generation, R, id=None):
    X = range(len(individuals_fitness_generation))
    Y = get_params_evolution(individuals_fitness_generation, R)

    Y_avg = [y[1] for y in Y]

    sub_plt = plt.subplot(2, 3, 6)
    sub_plt.grid(True)
    sub_plt.set_xlabel("Generations")
    sub_plt.set_ylabel("Parameters")
    sub_plt.plot(X, Y_avg, label=id)
    sub_plt.legend()


def plot_parameters_fittest_normalized_evolution(individuals_fitness_generation, R, GA_params, id=None):
    X = range(len(individuals_fitness_generation))
    _, Y = get_params_evolution(individuals_fitness_generation, R)
    _, fit = fitness_evolution(individuals_fitness_generation, R, GA_params)
    best = [np.argmax(fit_i) for fit_i in fit]
    Y = [y[best[i]]/27247905 for i, y in enumerate(Y)]

    sub_plt = plt.subplot(2, 3, 6)
    sub_plt.grid(True)
    sub_plt.set_xlabel("Generations")
    sub_plt.set_ylabel("x More Parameters")
    sub_plt.plot(X, Y, label=id)
    # sub_plt.plot(X, Y_max, linestyle='--')
    # sub_plt.legend()


if __name__ == "__main__":

    experiment = 'genetic_algorithm_multinode'
    plt.rcParams.update({'font.size': 6})

    # Accuracy-Time-Params
    # ids = [9143134805644978, 5921909506799964, 241492448257897, 3697276206940217, 2211146694894534, 8715929172344629, 9115600675480175,]
    # labels = ["w1=w2=w3=1/3", "w1=2/4; w23=1/4", "w1=3/5; w23=1/5", "w1=4/6; w23=1/6", "w1=5/7; w23=1/7", "w1=6/8; w23=1/8", "w1=8/10; w23=1/10"]

    # Accuracy-Time
    # ids = [6808594620290078, 1357102744523221, 2752090179980188, 2312341112477854, 1453813583148056, 1346534668312578, 3955787901367721]
    # labels = ["w1=1/5; w2=4/5", "w1=1/4; w2=3/4", "w1=1/3; w2=2/3", "w1=1/2; w2=1/2", "w1=2/3; w2=1/3", "w1=3/4; w2=1/4", "w1=4/5; w2=1/5"]

    # Accuracy-Params
    # ids = [2751770332018197, 8436681981021060, 7194612083294987, 5895615054661813, 2179484357724250, 9812369261926396, 6467063753693562]
    # labels = ["w1=1/5; w2=4/5", "w1=1/4; w2=3/4", "w1=1/3; w2=2/3", "w1=1/2; w2=1/2", "w1=2/3; w2=1/3", "w1=3/4; w2=1/4", "w1=4/5; w2=1/5"]

    # Accuray
    ids = [5161096127991005]
    labels = ["w1=1; w2=0; w3=0"]

    # 1) G.A. Chain ensembles
    GA_results_metadata_file = os.path.join(os.environ['FCM'],
                                            'Examples',
                                            'compute',
                                            experiment,
                                            'results',
                                            'metadata.json')

    plt.figure()

    # Get evaluation results from query
    for j, id in enumerate(ids):
        GA_res_loc = results_manager.get_results_by_id(GA_results_metadata_file, id)
        GA_params = results_manager.get_fieldval_by_id(GA_results_metadata_file, id, 'params')[0]
        individuals_fitness_generation = io.read_pickle(os.path.join(GA_res_loc, 'individuals_fitness_per_generation.pkl'))
        R = io.read_pickle(os.path.join(GA_res_loc, 'results_ensembles.pkl'))
        print('\n'.join(individuals_fitness_generation[-1][0]))

        # Plot performance of the Genetic Algorithm through generations
        plot_fitness_evolution(individuals_fitness_generation, R, GA_params, labels[j])
        plot_offspring_replacement_evolution(individuals_fitness_generation, labels[j])
        plot_repeated_individuals_evolution(individuals_fitness_generation, labels[j])

        # plot_accuracy_evolution(individuals_fitness_generation, R, labels[j])
        # plot_time_evolution(individuals_fitness_generation, R, labels[j])
        # plot_parameters_evolution(individuals_fitness_generation, R, labels[j])

        # plot_accuracy_normalized_evolution(individuals_fitness_generation, R, labels[j])
        # plot_time_normalized_evolution(individuals_fitness_generation, R, labels[j])
        # plot_parameters_normalized_evolution(individuals_fitness_generation, R, labels[j])

        plot_accuracy_fittest_normalized_evolution(individuals_fitness_generation, R, GA_params, labels[j])
        plot_time_fittest_normalized_evolution(individuals_fitness_generation, R, GA_params, labels[j])
        plot_parameters_fittest_normalized_evolution(individuals_fitness_generation, R, GA_params, labels[j])

    plt.show()

