import source.io_util as io
import source.genetic_algorithm.fitting_functions as fit_fun
import examples.paretto_front as front
import examples.metadata_manager_results as results_manager
import matplotlib.pyplot as plt
import numpy as np
import sys, os
import source

colors = [
                "#3bbfff",  # 0 Gray-ish blue
                "#2e97fd",  # 1 Light blue
                "#cccccc",  # 2 Light gray
                "#e1e130",  # 3 Light yellow,
                "#000080",  # 4 Navy blue
                "#8d1616",  # 5 Wine red
        ]


def subplot_acc_time(evaluation_results, ax):
    system_keys = [key for key in evaluation_results.keys() if len(evaluation_results[key].test) > 2]
    ref_keys = [key for key in evaluation_results.keys() if key not in system_keys]

    X = [evaluation_results[key].test['system'].time for key in system_keys]
    Y = [evaluation_results[key].test['system'].accuracy for key in system_keys]
    ax.scatter(X, Y,
               s=20,
               label='Chains',
               facecolor=colors[1],
               linewidth=0.1,
               edgecolor='black',
               marker='o')

    X = [evaluation_results[key].test['system'].time for key in ref_keys]
    Y = [evaluation_results[key].test['system'].accuracy for key in ref_keys]
    ax.scatter(X, Y,
               s=30,
               label='Single DNN',
               color='black',
               marker='P')


def subplot_acc_params(evaluation_results, ax):
    ax.set_xscale('log')

    system_keys = [key for key in evaluation_results.keys() if len(evaluation_results[key].test) > 2]
    ref_keys = [key for key in evaluation_results.keys() if key not in system_keys]

    X = [evaluation_results[key].test['system'].params*(4/1e6) for key in system_keys]
    Y = [evaluation_results[key].test['system'].accuracy for key in system_keys]
    ax.scatter(X, Y,
               s=20,
               label='Chains',
               facecolor=colors[1],
               linewidth=0.1,
               edgecolor='black',
               marker='o')

    X = [evaluation_results[key].test['system'].params*(4/1e6) for key in ref_keys]
    Y = [evaluation_results[key].test['system'].accuracy for key in ref_keys]
    ax.scatter(X, Y,
               s=30,
               label='Single DNN',
               color='black',
               marker='P')


def subplot_acc_ops(evaluation_results, ax):
    ax.set_xscale('log')

    system_keys = [key for key in evaluation_results.keys() if len(evaluation_results[key].test) > 2]
    ref_keys = [key for key in evaluation_results.keys() if key not in system_keys]

    X = [evaluation_results[key].test['system'].ops for key in system_keys]
    Y = [evaluation_results[key].test['system'].accuracy for key in system_keys]
    ax.scatter(X, Y,
               s=20,
               label='Chains',
               facecolor=colors[1],
               linewidth=0.1,
               edgecolor='black',
               marker='o')

    X = [evaluation_results[key].test['system'].ops for key in ref_keys]
    Y = [evaluation_results[key].test['system'].accuracy for key in ref_keys]
    ax.scatter(X, Y,
               s=30,
               label='Single DNN',
               color='black',
               marker='P')


def subplot_acc_time_paretto_front(evaluation_results, ax, color='black', label=""):
    pareto_solutions = front.get_front_time_accuracy(evaluation_results, phase="test")
    pareto_solutions_sorted = front.sort_results_by_accuracy(pareto_solutions, phase="test")

    X = [eval_result.test['system'].time for (key, eval_result) in pareto_solutions_sorted]
    Y = [eval_result.test['system'].accuracy for (key, eval_result) in pareto_solutions_sorted]

    X[:0] = X[::2] = X[1::2] = X[:]  # Duplicate each element
    Y[:0] = Y[::2] = Y[1::2] = Y[:]  # Duplicate each element

    del X[0]
    X+=[20]

    ax.plot(X, Y,
            linestyle="-",
            label=label,
            color=color,
            linewidth=2,
            )


def subplot_acc_params_paretto_front(evaluation_results, ax, color='black', label=""):
    pareto_solutions = front.get_front_params_accuracy(evaluation_results, phase="test")
    pareto_solutions_sorted = front.sort_results_by_params(pareto_solutions, phase="test")

    X = [eval_result.test['system'].params*(4/1e6) for (key, eval_result) in pareto_solutions_sorted]
    Y = [eval_result.test['system'].accuracy for (key, eval_result) in pareto_solutions_sorted]

    Y[:0] = Y[::2] = Y[1::2] = Y[:]  # Duplicate each element
    X[:0] = X[::2] = X[1::2] = X[:]

    del X[0]
    X+=[1e10]

    ax.plot(X, Y,
            linestyle='-',
            label=label,
            color=color,
            linewidth=2,
            markersize=50,
           )


def subplot_acc_ops_paretto_front(evaluation_results, ax, color='black', label=""):
    pareto_solutions = front.get_front_ops_accuracy(evaluation_results, phase="test")
    pareto_solutions_sorted = front.sort_results_by_ops(pareto_solutions, phase="test")

    X = [eval_result.test['system'].ops for (key, eval_result) in pareto_solutions_sorted]
    Y = [eval_result.test['system'].accuracy for (key, eval_result) in pareto_solutions_sorted]

    Y[:0] = Y[::2] = Y[1::2] = Y[:]  # Duplicate each element
    X[:0] = X[::2] = X[1::2] = X[:]

    del X[0]
    X+=[1e10]

    ax.plot(X, Y,
            linestyle='-',
            label=label,
            color=color,
            linewidth=2,
            markersize=50,
           )


def subplot_most_fit_chain(evaluation_results, ax):

    # Most accurate single DNN
    evaluation_results_singleDNN = dict([(key, result) for key, result in evaluation_results.items() if len(result.test) < 3])
    best_DNN = front.sort_results_by_accuracy(evaluation_results_singleDNN, phase="test")[-1]
    accuracy_dnn = best_DNN[1].test['system'].accuracy

    # Fittest chain
    evaluation_results_list = [result for key, result in evaluation_results.items()]
    fitness = np.array(fit_fun.f1_time_param_penalization(evaluation_results_list, phase="test"))*-1
    ordered = np.argsort(fitness)
    best = 0
    for index in ordered:
        if evaluation_results_list[index].test['system'].accuracy >= accuracy_dnn:
            best = index
            print("Found!")
            break

    # Accuracy vs Params
    x = evaluation_results_list[best].test['system'].params * (4/1e6)
    y = evaluation_results_list[best].test['system'].accuracy
    ax[1].scatter(x, y,
               s=65,
               label='Fittest chain',
               facecolor=colors[3],
               linewidth=0.5,
               edgecolor='black',
               marker='o',
               zorder=10)

    # Accuracy vs Time
    x = evaluation_results_list[best].test['system'].time
    y = evaluation_results_list[best].test['system'].accuracy
    ax[0].scatter(x, y,
                s=65,
                label='Fittest chain',
                facecolor=colors[3],
                linewidth=0.5,
                edgecolor='black',
                marker='o',
                zorder=10)

    # Accuracy vs Time
    x = evaluation_results_list[best].test['system'].ops
    y = evaluation_results_list[best].test['system'].accuracy
    ax[2].scatter(x, y,
                s=65,
                label='Fittest chain',
                facecolor=colors[3],
                linewidth=0.5,
                edgecolor='black',
                marker='o',
                zorder=10)


def plot_acc_time_param_tradeoff(evaluation_results):
    fig, ax = plt.subplots(1, 3, tight_layout=True)

    ax[0].grid(linewidth=0.4)
    ax[0].set_xlabel('Time (s) on GPU in test set evaluation', fontsize=12)
    ax[0].set_ylabel('Test accuracy', fontsize=12)
    ax[0].set_ylim(bottom=0.39, top=0.56)
    ax[0].set_xlim(right=12)

    ax[1].grid(linewidth=0.4)
    ax[1].set_xlabel('Memory footprint (MBytes)', fontsize=12)
    ax[1].set_ylabel('Test accuracy', fontsize=12)
    ax[1].set_ylim(bottom=0.39, top=0.56)
    ax[1].set_xlim(left=1e0, right=1e3)

    ax[2].grid(linewidth=0.4)
    ax[2].set_xlabel('Number of operations in test set evaluation', fontsize=12)
    ax[2].set_ylabel('Test accuracy', fontsize=12)
    ax[2].set_ylim(bottom=0.39, top=0.56)
    ax[2].set_xlim(left=3e7, right=1e10)

    ax.flat[1].set_title('EARN on Caltech-256')

    # Print pareto frontiers on top
    subplot_acc_time_paretto_front(evaluation_results, ax[0], color=colors[5], label="Chains Pareto frontier")
    subplot_acc_params_paretto_front(evaluation_results, ax[1], color=colors[5], label="Chains Pareto frontier")
    subplot_acc_ops_paretto_front(evaluation_results, ax[2], color=colors[5], label="Chains Pareto frontier")
    # Print pareto frontiers for single DNN on top
    evaluation_results_DNN = dict([(key, result) for key, result in evaluation_results.items() if len(result.test) < 3])
    subplot_acc_time_paretto_front(evaluation_results_DNN, ax[0], color='black', label="Single DNN Pareto frontier")
    subplot_acc_params_paretto_front(evaluation_results_DNN, ax[1], color='black', label="Single DNN Pareto frontier")
    subplot_acc_ops_paretto_front(evaluation_results_DNN, ax[2], color='black', label="Single DNN Pareto frontier")
    # Fill axes with solutions
    subplot_acc_time(evaluation_results, ax[0])
    subplot_acc_params(evaluation_results, ax[1])
    subplot_acc_ops(evaluation_results, ax[2])
    # Highlight fittest chain
    subplot_most_fit_chain(evaluation_results, ax)

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    return ax


if __name__ == "__main__":
    experiment = 'chain_genetic_algorithm'
    id = 54099516605262
    experiment_dir = os.path.join(os.environ['FCM'], 'examples', 'compute', experiment)
    meta_data_file = os.path.join(experiment_dir, 'results', 'metadata.json')
    earn_evaluations_location = results_manager.get_results_by_id(meta_data_file, id)

    sys.modules['Source'] = source  # Due to refactor of the module
    earn_evaluations = io.read_pickle(earn_evaluations_location)
    ax = plot_acc_time_param_tradeoff(earn_evaluations)

    """
    brute_force_location = os.path.join(os.environ['FCM'],
                                        "examples/compute/fully_connected_chain/results/sota_models_caltech256-32-dev_validation/3971529594480967/results_ensembles.pkl")
    brute_force_results = io.read_pickle(brute_force_location)
    subplot_acc_time_paretto_front(brute_force_results, ax[0], color='green')
    subplot_acc_params_paretto_front(brute_force_results, ax[1], color='green')
    ax[0].legend()
    ax[1].legend()
    """

    plt.show()

