from Examples import metadata_manager_results as results_manager
from Source.genetic_algorithm.fitting_functions import f2_time_param_penalization
from Source.genetic_algorithm.fitting_functions import make_limits_dict, update_limit_dict
from Source import io_util as io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os


ref_NN = None
limit = make_limits_dict()
phase = ""


def select_reference_NN(R):
    global ref_NN
    global limit
    models = dict([(k, r) for k, r in R.items() if len(r.val) < 3])

    model_ids = list(models.keys())
    best = np.argmax([R[id].test['system'].accuracy if phase == "test" else R[id].val['system'].accuracy for id in model_ids])
    ref_NN = models[model_ids[best]]
    update_limit_dict(limit, models, phase=phase)
    print(limit)



def fitness_evolution(individuals_fitness_generation, R, GA_params):
    fit_stats = []
    fit_plain = []

    for generation in individuals_fitness_generation:

        # Gathering results
        ids = generation[0]
        R_ids = [R[id] for id in ids]

        # Fitness value of results
        fit_ids = f2_time_param_penalization(R_ids, GA_params['a'], limit, phase=phase)
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
        acc_ids = [r.test['system'].accuracy if phase == "test" else r.val['system'].accuracy for r in R_ids]
        acc_plain.append(np.array(acc_ids))
        acc += [(max(acc_ids), sum(acc_ids) / len(acc_ids), min(acc_ids))]
    return acc, acc_plain


def get_time_evolution(individuals_fitness_generation, R):
    Y = []
    Y_plain = []
    for generation in individuals_fitness_generation:
        ids = generation[0]
        R_ids = [R[id] for id in ids]
        time_gen = [r.test['system'].time if phase == "test" else r.val['system'].time for r in R_ids]
        Y_plain.append(np.array(time_gen))
        Y += [(max(time_gen), sum(time_gen) / len(time_gen), min(time_gen))]
    return Y, Y_plain


def get_params_evolution(individuals_fitness_generation, R):
    Y = []
    Y_plain = []
    for generation in individuals_fitness_generation:
        ids = generation[0]
        R_ids = [R[id] for id in ids]
        params_gen = [r.test['system'].params if phase == "test" else r.val['system'].params for r in R_ids]
        Y_plain.append(params_gen)
        Y += [(max(params_gen), sum(params_gen) / len(params_gen), min(params_gen))]
    return Y, Y_plain


def get_num_classifiers_evolution(individuals_fitness_generation, R):
    Y = []
    Y_plain = []
    for generation in individuals_fitness_generation:
        ids = generation[0]
        R_ids = [R[id] for id in ids]
        params_gen = []
        for r in R_ids:
            if phase =="test":
                classifiers = [c for c in r.test.keys() if "trigger" not in c and "erger" not in c]
                params_gen.append(len(classifiers) - 1)
            else:
                classifiers = [c for c in r.val.keys() if "trigger" not in c and "erger" not in c]
                params_gen.append(len(classifiers) - 1)
        Y_plain.append(params_gen)
        Y += [(max(params_gen), sum(params_gen) / len(params_gen), min(params_gen))]
    return Y, Y_plain


def plot_fitness_evolution(X, Y, label=None, color=None, linestyle=None):
    Y_avg = [y[1] for y in Y]

    sub_plt = plt.subplot(2, 3, 1)
    sub_plt.grid(True)
    sub_plt.set_xlabel("Generations")
    sub_plt.set_ylabel("Test Average Fitness Value")
    sub_plt.plot(X, Y_avg, label=label, color=color, linestyle=linestyle)


def plot_fitness_fittest_evolution(X, Y, label=None, color=None, linestyle=None):

    sub_plt = plt.subplot(2, 3, 1)
    sub_plt.grid(True)
    sub_plt.set_xlabel("Generations")
    sub_plt.set_ylabel("Fittest Fitness Value in " + phase)
    sub_plt.plot(X, Y, label=label, color=color, linestyle=linestyle)


def plot_offspring_replacement_evolution(X , Y, label=None, color=None, linestyle="-"):
    sub_plt = plt.subplot(2, 3, 2)
    sub_plt.grid(True)
    sub_plt.set_xlabel("Generations")
    sub_plt.set_ylabel("Offspring survival rate")
    sub_plt.set_ylim(bottom=0, top=1)
    sub_plt.plot(X, Y, label=label, linestyle=linestyle, color=color)


def plot_repeated_individuals_evolution(X, Y, label=None, color=None, linestyle="-"):
    sub_plt = plt.subplot(2, 3, 3)
    sub_plt.grid(True)
    sub_plt.set_xlabel("Generations")
    sub_plt.set_ylabel("Ratio Unique")
    sub_plt.set_ylim(bottom=0, top=1)
    sub_plt.plot(X, Y, label=label, linestyle=linestyle, color=color)
    sub_plt.legend()


def plot_accuracy_evolution(X, Y, label=None, color=None, linestyle="-"):
    Y_avg = [y[1] for y in Y]

    sub_plt = plt.subplot(2, 3, 4)
    sub_plt.grid(True)
    sub_plt.set_xlabel("Generations")
    sub_plt.set_ylabel(phase + "accuracy")
    sub_plt.plot(X, Y_avg, label=label, linestyle=linestyle, color=color)


def plot_accuracy_fittest_evolution(X, Y, label=None, color=None, linestyle="-"):
    """
    X = range(len(individuals_fitness_generation))
    _, Y = get_accuracy_evolution(individuals_fitness_generation, R)
    _, fit = fitness_evolution(individuals_fitness_generation, R, GA_params)

    best = [np.argmax(fit_i) for fit_i in fit]
    Y = [y[best[i]] for i, y in enumerate(Y)]
    """

    sub_plt = plt.subplot(2, 3, 4)
    sub_plt.grid(True)
    sub_plt.set_xlabel("Generations")
    sub_plt.set_ylabel(phase + " accuracy")
    sub_plt.plot(X, Y, label=label, linestyle=linestyle, color=color)


def plot_accuracy_fittest_normalized_evolution(X, Y, label=None, color=None, linestyle="-"):
    """
    X = range(len(individuals_fitness_generation))
    _, Y = get_accuracy_evolution(individuals_fitness_generation, R)
    _, fit = fitness_evolution(individuals_fitness_generation, R, GA_params)

    best = [np.argmax(fit_i) for fit_i in fit]
    Y = [(y[best[i]]-ref_NN.test['system'].accuracy)*100 for i, y in enumerate(Y)]
    """
    sub_plt = plt.subplot(2, 3, 4)
    sub_plt.grid(True)
    sub_plt.set_xlabel("Generations")
    sub_plt.set_ylabel("Increase "+ phase +" accuracy (%)")
    sub_plt.plot(X, Y, label=label, linestyle=linestyle, color=color)


def plot_time_evolution(X, Y, label=None, color=None, linestyle="-"):
    Y_avg = [y[1] for y in Y]

    sub_plt = plt.subplot(2, 3, 5)
    sub_plt.grid(True)
    sub_plt.set_xlabel("Generations")
    sub_plt.set_ylabel("Expected inference time")
    sub_plt.plot(X, Y_avg, label=label, linestyle=linestyle, color=color)


def plot_time_fittest_normalized_evolution(X, Y, label=None, color=None, linestyle="-"):
    """
    X = range(len(individuals_fitness_generation))
    _, Y = get_time_evolution(individuals_fitness_generation, R)
    _, fit = fitness_evolution(individuals_fitness_generation, R, GA_params)
    best = [np.argmax(fit_i) for fit_i in fit]
    Y = [ref_NN.test['system'].time/y[best[i]] for i, y in enumerate(Y)]
    """
    sub_plt = plt.subplot(2, 3, 5)
    sub_plt.grid(True)
    sub_plt.set_xlabel("Generations")
    sub_plt.set_ylabel("Speedup")
    sub_plt.plot(X, Y, label=label, linestyle=linestyle, color=color)


def plot_parameters_evolution(X, Y, label=None, color=None, linestyle="-"):
    Y_avg = [y[1] for y in Y]

    sub_plt = plt.subplot(2, 3, 6)
    sub_plt.grid(True)
    sub_plt.set_xlabel("Generations")
    sub_plt.set_ylabel("Parameters")
    sub_plt.plot(X, Y_avg, label=label, linestyle=linestyle, color=color)


def plot_parameters_fittest_normalized_evolution(X, Y, label=None, color=None, linestyle="-"):
    """
    X = range(len(individuals_fitness_generation))
    _, Y = get_params_evolution(individuals_fitness_generation, R)
    _, fit = fitness_evolution(individuals_fitness_generation, R, GA_params)
    best = [np.argmax(fit_i) for fit_i in fit]
    Y = [y[best[i]]/ref_NN.test['system'].params for i, y in enumerate(Y)]
    """
    sub_plt = plt.subplot(2, 3, 6)
    sub_plt.grid(True)
    sub_plt.set_xlabel("Generations")
    sub_plt.set_ylabel("x More Parameters")
    sub_plt.plot(X, Y, label=label, linestyle=linestyle, color=color)


def plot_complexity_fittest_evolution(X, Y, label=None, color=None, linestyle="-"):
    """
    X = range(len(individuals_fitness_generation))
    _, Y = get_params_evolution(individuals_fitness_generation, R)
    _, fit = fitness_evolution(individuals_fitness_generation, R, GA_params)
    best = [np.argmax(fit_i) for fit_i in fit]
    Y = [y[best[i]]/ref_NN.test['system'].params for i, y in enumerate(Y)]
    """
    sub_plt = plt.subplot(2, 3, 2)
    sub_plt.grid(True)
    sub_plt.set_xlabel("Generations")
    sub_plt.set_ylabel("Number of classifiers")
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

    ids = [[4604161110947493, 1083678468052179, 714527043472789, 1888476320409151, 2446234969539517],  # w1=0.7 (caltech, cifar100, cifar10, svhn, stl10)
            [7282656004905524, 2008971533382462, 7254994050071607, 8818109826887893, 2784611815264723],  # w1=0.8 (caltech, cifar100, cifar10, svhn, stl10)
            [1167621589601369, 1967022650296029, 1977507596855284, 731196218074309, 12666180908584]]  # w1=0.9 (caltech, cifar100, cifar10, svhn, stl10)

    ids = [[7053119233470466, 6924025530716953, 9975332911346946, 9564212798404591, 2996958336299889],  # w1=0.7 (caltech, cifar100, cifar10, svhn, stl10)
           [3028012080632813, 3815604751209093, 5694898984571698, 482183001946958, 6172208740384491],  # w1=0.8 (caltech, cifar100, cifar10, svhn, stl10)
           [2051217125919466, 9137831047978630, 8677042312622802, 5570096773173367, 1978330328399986]]  # w1=0.9 (caltech, cifar100, cifar10, svhn, stl10)

    ids = [[405176288657413],
           [9862484120693946]]

    phase = "val"
    labels = ["Crossover_2_pm=0.7", "Crossover_1_pm=0.9", "crossover_1_pm=0.9", "crossover_1", "crossover_1_pm=0.8"]
    line_style = ['-']*2 + ['--']*3
    cmap = cm.get_cmap('jet')
    colors = cmap(np.linspace(0, 1.0, 2))
    colors = np.append(colors, cmap(np.linspace(0, 1.0, 3)), axis=0)
    ##################################################################

    plt.figure()
    for j, id in enumerate(ids):

        Y_acc_avg = np.empty((0, 200))
        Y_time_avg = np.empty((0, 200))
        Y_params_avg = np.empty((0, 200))
        Y_fitness_avg = np.empty((0, 200))
        Y_num_classifiers_avg = np.empty((0, 200))
        X = range(200)

        for sub_id in id:
            GA_res_loc = results_manager.get_results_by_id(GA_results_metadata_file, sub_id)
            GA_params = results_manager.get_fieldval_by_id(GA_results_metadata_file, sub_id, 'params')[0]
            individuals_fitness_generation = io.read_pickle(os.path.join(GA_res_loc, 'individuals_fitness_per_generation.pkl'))
            R = io.read_pickle(os.path.join(GA_res_loc, 'results_ensembles.pkl'))

            select_reference_NN(R)

            Y_acc_stats, Y_acc_all = get_accuracy_evolution(individuals_fitness_generation, R)
            Y_time_stats, Y_time_all = get_time_evolution(individuals_fitness_generation, R)
            Y_params_stats, Y_params_all = get_params_evolution(individuals_fitness_generation, R)
            Y_fitness_stats, Y_fitness_all = fitness_evolution(individuals_fitness_generation, R, GA_params)
            Y_num_classifier_stats, Y_num_classifiers_all = get_num_classifiers_evolution(individuals_fitness_generation, R)

            best = [np.argmax(fit_i) for fit_i in Y_fitness_all]
            Y_fitness = [y[0] for y in Y_fitness_stats]
            Y_acc = [(y[best[i]]-ref_NN.test['system'].accuracy)*100 for i, y in enumerate(Y_acc_all)]
            Y_time = [ref_NN.test['system'].time/y[best[i]] for i, y in enumerate(Y_time_all)]
            Y_params = [y[best[i]]/ref_NN.test['system'].params for i, y in enumerate(Y_params_all)]
            Y_num_classifiers = [y[best[i]] for i, y in enumerate(Y_num_classifiers_all)]

            Y_acc_avg = np.vstack((Y_acc_avg, Y_acc))
            Y_time_avg = np.vstack((Y_time_avg, Y_time))
            Y_params_avg = np.vstack((Y_params_avg, Y_params))
            Y_fitness_avg = np.vstack((Y_fitness_avg, Y_fitness))
            Y_num_classifiers_avg = np.vstack((Y_num_classifiers_avg, Y_num_classifiers))

        Y_acc_avg = np.average(Y_acc_avg, axis=0)
        Y_time_avg = np.average(Y_time_avg, axis=0)
        Y_params_avg = np.average(Y_params_avg, axis=0)
        Y_fitness_avg = np.average(Y_fitness_avg, axis=0)
        Y_num_classifiers_avg = np.average(Y_num_classifiers_avg, axis=0)

        plot_fitness_fittest_evolution(X, Y_fitness_avg, labels[j], colors[j], line_style[j])
        plot_complexity_fittest_evolution(X, Y_num_classifiers_avg, labels[j], colors[j], line_style[j])
        #plot_offspring_replacement_evolution(X, percentage_replacement_offspring(individuals_fitness_generation), labels[j], colors[j], line_style[j])
        #plot_repeated_individuals_evolution(X, percentage_repeated_individuals(individuals_fitness_generation), labels[j], colors[j], line_style[j])
        plot_accuracy_fittest_normalized_evolution(X, Y_acc_avg, labels[j], colors[j], line_style[j])
        plot_time_fittest_normalized_evolution(X, Y_time_avg, labels[j], colors[j], line_style[j])
        plot_parameters_fittest_normalized_evolution(X, Y_params_avg, labels[j], colors[j], line_style[j])
    plt.legend(labels)
    plt.show()
