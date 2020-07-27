from Examples.study import metadata_manager_results as results_manager
from Source.genetic_algorithm.fitting_functions import f2_time_param_penalization
from Source.genetic_algorithm.fitting_functions import make_limits_dict
from Source import io_util as io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os


ref_NN = None
limit = make_limits_dict()
mode = "test"


def select_reference_NN(R):
    global ref_NN
    global limit
    models = dict([(k, r) for k, r in R.items() if len(r.test) < 3])

    model_ids = list(models.keys())
    best = np.argmax([R[id].test['system'].accuracy if mode == "test" else R[id].val['system'].accuracy for id in model_ids])
    ref_NN = models[model_ids[best]]

    limit['max_accuracy'] = max([R[id].test['system'].accuracy if mode == "test" else R[id].val['system'].accuracy for id in model_ids])
    limit['min_accuracy'] = min([R[id].test['system'].accuracy if mode == "test" else R[id].val['system'].accuracy for id in model_ids])
    limit['max_time'] = max([R[id].test['system'].time if mode == "test" else R[id].val['system'].time for id in model_ids])
    limit['min_time'] = min([R[id].test['system'].time if mode == "test" else R[id].val['system'].time for id in model_ids])
    limit['max_params'] = max([R[id].test['system'].params if mode == "test" else R[id].val['system'].params for id in model_ids])
    limit['min_params'] = min([R[id].test['system'].params if mode == "test" else R[id].val['system'].params for id in model_ids])


def fitness_evolution(individuals_fitness_generation, R, GA_params):
    fit_stats = []
    fit_plain = []

    for generation in individuals_fitness_generation:

        # Gathering results
        ids = generation[0]
        R_ids = [R[id] for id in ids]

        # Fitness value of results
        fit_ids = f2_time_param_penalization(R_ids, GA_params['a'], limit, phase=mode)
        fit_plain.append(np.array(fit_ids[:]))
        fit_stats += [(max(fit_ids[:]), sum(fit_ids[:])/len(fit_ids[:]), min(fit_ids[:]))]

    return fit_stats, fit_plain


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
        acc_ids = [r.test['system'].accuracy if mode == "test" else r.val['system'].accuracy for r in R_ids]
        acc_plain.append(np.array(acc_ids))
        acc += [(max(acc_ids), sum(acc_ids) / len(acc_ids), min(acc_ids))]
    return acc, acc_plain


def get_time_evolution(individuals_fitness_generation, R):
    Y = []
    Y_plain = []
    for generation in individuals_fitness_generation:
        ids = generation[0]
        R_ids = [R[id] for id in ids]
        time_gen = [r.test['system'].time if mode == "test" else r.val['system'].time for r in R_ids]
        Y_plain.append(np.array(time_gen))
        Y += [(max(time_gen), sum(time_gen) / len(time_gen), min(time_gen))]
    return Y, Y_plain


def get_params_evolution(individuals_fitness_generation, R):
    Y = []
    Y_plain = []
    for generation in individuals_fitness_generation:
        ids = generation[0]
        R_ids = [R[id] for id in ids]
        params_gen = [r.test['system'].params if mode == "test" else r.val['system'].params for r in R_ids]
        Y_plain.append(params_gen)
        Y += [(max(params_gen), sum(params_gen) / len(params_gen), min(params_gen))]
    return Y, Y_plain


def plot_fitness_evolution(individuals_fitness_generation, R, GA_params, label=None, color=None, linestyle=None):
    X = range(len(individuals_fitness_generation))
    Y, _= fitness_evolution(individuals_fitness_generation, R, GA_params)
    Y_average = [y[1] for y in Y]

    sub_plt = plt.subplot(2, 3, 1)
    sub_plt.grid(True)
    sub_plt.set_xlabel("Generations")
    sub_plt.set_ylabel("Test Average Fitness Value")
    sub_plt.plot(X, Y_average, label=label, color=color, linestyle=linestyle)


def plot_fitness_fittest_evolution(individuals_fitness_generation, R, GA_params, label=None, color=None, linestyle=None):
    X = range(len(individuals_fitness_generation))
    Y, _= fitness_evolution(individuals_fitness_generation, R, GA_params)
    Y_max = [y[0] for y in Y]

    sub_plt = plt.subplot(2, 3, 1)
    sub_plt.grid(True)
    sub_plt.set_xlabel("Generations")
    sub_plt.set_ylabel("Fittest Fitness Value in " + mode)
    sub_plt.plot(X, Y_max, label=label, color=color, linestyle=linestyle)


def plot_offspring_replacement_evolution(individuals_fitness_generation, label=None, color=None, linestyle="-"):
    X = range(len(individuals_fitness_generation))
    Y = percentage_replacement_offspring(individuals_fitness_generation)

    sub_plt = plt.subplot(2, 3, 2)
    sub_plt.grid(True)
    sub_plt.set_xlabel("Generations")
    sub_plt.set_ylabel("Offspring survival rate")
    sub_plt.set_ylim(bottom=0, top=1)
    sub_plt.plot(X, Y, label=label, linestyle=linestyle, color=color)


def plot_repeated_individuals_evolution(individuals_fitness_generation, label=None, color=None, linestyle="-"):
    X = range(len(individuals_fitness_generation))
    Y = percentage_repeated_individuals(individuals_fitness_generation)

    sub_plt = plt.subplot(2, 3, 3)
    sub_plt.grid(True)
    sub_plt.set_xlabel("Generations")
    sub_plt.set_ylabel("Ratio Unique")
    sub_plt.set_ylim(bottom=0, top=1)
    sub_plt.plot(X, Y, label=label, linestyle=linestyle, color=color)
    sub_plt.legend()


def plot_accuracy_evolution(individuals_fitness_generation, R, label=None, color=None, linestyle="-"):
    X = range(len(individuals_fitness_generation))
    Y, _ = get_accuracy_evolution(individuals_fitness_generation, R)
    Y_avg = [y[1] for y in Y]

    sub_plt = plt.subplot(2, 3, 4)
    sub_plt.grid(True)
    sub_plt.set_xlabel("Generations")
    sub_plt.set_ylabel(mode + "accuracy")
    sub_plt.plot(X, Y_avg, label=label, linestyle=linestyle, color=color)


def plot_accuracy_fittest_evolution(individuals_fitness_generation, R, GA_params, label=None, color=None, linestyle="-"):
    X = range(len(individuals_fitness_generation))
    _, Y = get_accuracy_evolution(individuals_fitness_generation, R)
    _, fit = fitness_evolution(individuals_fitness_generation, R, GA_params)
    best = [np.argmax(fit_i) for fit_i in fit]
    Y = [y[best[i]] for i, y in enumerate(Y)]

    sub_plt = plt.subplot(2, 3, 4)
    sub_plt.grid(True)
    sub_plt.set_xlabel("Generations")
    sub_plt.set_ylabel(mode + " accuracy")
    sub_plt.plot(X, Y, label=label, linestyle=linestyle, color=color)


def plot_accuracy_fittest_normalized_evolution(individuals_fitness_generation, R, GA_params, label=None, color=None, linestyle="-"):
    X = range(len(individuals_fitness_generation))
    _, Y = get_accuracy_evolution(individuals_fitness_generation, R)
    _, fit = fitness_evolution(individuals_fitness_generation, R, GA_params)
    best = [np.argmax(fit_i) for fit_i in fit]
    Y = [(y[best[i]]-ref_NN.test['system'].accuracy)*100 for i, y in enumerate(Y)]

    sub_plt = plt.subplot(2, 3, 4)
    sub_plt.grid(True)
    sub_plt.set_xlabel("Generations")
    sub_plt.set_ylabel("Increase "+mode+" accuracy (%)")
    sub_plt.plot(X, Y, label=label, linestyle=linestyle, color=color)


def plot_time_evolution(individuals_fitness_generation, R, label=None, color=None, linestyle="-"):
    X = range(len(individuals_fitness_generation))
    Y = get_time_evolution(individuals_fitness_generation, R)
    Y_avg = [y[1] for y in Y]

    sub_plt = plt.subplot(2, 3, 5)
    sub_plt.grid(True)
    sub_plt.set_xlabel("Generations")
    sub_plt.set_ylabel("Expected inference time")
    sub_plt.plot(X, Y_avg, label=label, linestyle=linestyle, color=color)


def plot_time_fittest_normalized_evolution(individuals_fitness_generation, R, GA_params, label=None, color=None, linestyle="-"):
    X = range(len(individuals_fitness_generation))
    _, Y = get_time_evolution(individuals_fitness_generation, R)
    _, fit = fitness_evolution(individuals_fitness_generation, R, GA_params)
    best = [np.argmax(fit_i) for fit_i in fit]
    Y = [ref_NN.test['system'].time/y[best[i]] for i, y in enumerate(Y)]

    sub_plt = plt.subplot(2, 3, 5)
    sub_plt.grid(True)
    sub_plt.set_xlabel("Generations")
    sub_plt.set_ylabel("Speedup")
    sub_plt.plot(X, Y, label=label, linestyle=linestyle, color=color)


def plot_parameters_evolution(individuals_fitness_generation, R, label=None, color=None, linestyle="-"):
    X = range(len(individuals_fitness_generation))
    Y = get_params_evolution(individuals_fitness_generation, R)

    Y_avg = [y[1] for y in Y]

    sub_plt = plt.subplot(2, 3, 6)
    sub_plt.grid(True)
    sub_plt.set_xlabel("Generations")
    sub_plt.set_ylabel("Parameters")
    sub_plt.plot(X, Y_avg, label=label, linestyle=linestyle, color=color)


def plot_parameters_fittest_normalized_evolution(individuals_fitness_generation, R, GA_params, label=None, color=None, linestyle="-"):
    X = range(len(individuals_fitness_generation))
    _, Y = get_params_evolution(individuals_fitness_generation, R)
    _, fit = fitness_evolution(individuals_fitness_generation, R, GA_params)
    best = [np.argmax(fit_i) for fit_i in fit]
    Y = [y[best[i]]/ref_NN.test['system'].params for i, y in enumerate(Y)]

    sub_plt = plt.subplot(2, 3, 6)
    sub_plt.grid(True)
    sub_plt.set_xlabel("Generations")
    sub_plt.set_ylabel("x More Parameters")
    sub_plt.plot(X, Y, label=label, linestyle=linestyle, color=color)


if __name__ == "__main__":

    ################### Plot configurations #########################
    experiment = 'bagging_boosting_of_chains_GA'
    plt.rcParams.update({'font.size': 12})
    GA_results_metadata_file = os.path.join(os.environ['FCM'],
                                            'Examples',
                                            'compute',
                                            experiment,
                                            'results',
                                            'metadata.json')
    mode = "test"
    ids = [1967022650296029]
    labels = ["caltech", "cifar100", "cifar10", "svhn", "stl10"] * 2
    line_style = ['-']*5 + ['--']*3
    cmap = cm.get_cmap('jet')
    colors = cmap(np.linspace(0, 1.0, 3))
    colors = np.append(colors, cmap(np.linspace(0, 1.0, 3)), axis=0)
    ##################################################################

    plt.figure()
    for j, id in enumerate(ids):

        GA_res_loc = results_manager.get_results_by_id(GA_results_metadata_file, id)
        GA_params = results_manager.get_fieldval_by_id(GA_results_metadata_file, id, 'params')[0]
        individuals_fitness_generation = io.read_pickle(os.path.join(GA_res_loc, 'individuals_fitness_per_generation.pkl'))
        R = io.read_pickle(os.path.join(GA_res_loc, 'results_ensembles.pkl'))

        if ref_NN is None:
            plt.suptitle("EARN execution on: " + GA_params['dataset'].split('_')[2].split('-')[0], fontsize=16)
            select_reference_NN(R)

        plot_fitness_fittest_evolution(individuals_fitness_generation, R, GA_params, labels[j], colors[j], line_style[j])
        plot_offspring_replacement_evolution(individuals_fitness_generation, labels[j], colors[j], line_style[j])
        plot_repeated_individuals_evolution(individuals_fitness_generation, labels[j], colors[j], line_style[j])

        plot_accuracy_fittest_normalized_evolution(individuals_fitness_generation, R, GA_params, labels[j], colors[j], line_style[j])
        plot_time_fittest_normalized_evolution(individuals_fitness_generation, R, GA_params, labels[j], colors[j], line_style[j])
        plot_parameters_fittest_normalized_evolution(individuals_fitness_generation, R, GA_params, labels[j], colors[j], line_style[j])

    plt.show()
