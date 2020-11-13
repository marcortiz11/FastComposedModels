from Source.genetic_algorithm.fitting_functions import normalize_error_time_params as f
from Source.genetic_algorithm.fitting_functions import make_limits_dict, update_limit_dict
from Source.genetic_algorithm.moo import hvolume
from Examples import metadata_manager_results as results_manager
import matplotlib.pyplot as plt
import Source.io_util as io
import matplotlib.cm as cm
import numpy as np
import hvwfg
import os


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

    ids = results_manager.get_ids_by_fieldval(GA_results_metadata_file, "params",
                                              {"experiment": "main_NGSA2.py",
                                               "dataset": "sota_models_caltech256-32-dev_validation",
                                               "iterations": 50,
                                               "a": [1, 1, 1]})
    # ids = [2863239621186222, 3508166830298376]

    phase = "test"
    labels = ["EARN pm=0.8", "GP pm=0.8", "EARN pm=0.5", "GP pm=0.5"]
    line_style = [':', '-.', '--', '-']
    cmap = cm.get_cmap('jet')
    colors = cmap(np.linspace(0, 1.0, len(ids)))
    ##################################################################

    plt.figure()
    limits = make_limits_dict()
    hvol_all = []
    params_all = []  # List of tuple (population, offspring, pm, rm, k)

    for j, id in enumerate(ids):
        GA_res_loc = results_manager.get_results_by_id(GA_results_metadata_file, id)
        GA_params = results_manager.get_fieldval_by_id(GA_results_metadata_file, id, "params")
        params_all.append((GA_params[0]['population'], GA_params[0]['offspring'], GA_params[0]['pm']))
        individuals_fitness_generation = io.read_pickle(
            os.path.join(GA_res_loc, 'individuals_fitness_per_generation.pkl'))
        R = io.read_pickle(os.path.join(GA_res_loc, 'results_ensembles.pkl'))
        R_models = dict([(k, r) for k, r in R.items() if len(r.val if phase == "val" else r.test) < 3])
        update_limit_dict(limits, R_models, phase)

        # Reference point
        ref = np.array([1., 1., 1.])
        Y_hvolume = []

        for i, gen in enumerate(individuals_fitness_generation):
            Rgen = [R[g] for g in gen[0]]
            obj = f(Rgen, limits,  phase)
            obj = obj[:, GA_params[0]['a'] == 1]
            Y_hvolume.append(hvolume(obj, ref))

        plt.plot(Y_hvolume, label=id, color=colors[j])
        hvol_all.append(Y_hvolume[-1])

    plt.legend()
    plt.show()

    # Correlation between variables
    plt.figure(2)
    hvol_all = np.array(hvol_all)
    population = np.array([param[0] for param in params_all])
    offspring = np.array([param[1] for param in params_all])
    pmutation = np.array([param[2] for param in params_all])
    # rmutation = [param[3] for param in params_all]

    x = []
    y = []
    plt.subplot(2, 2, 1)
    for p in np.unique(population):
        ids = np.where(population == p)[0]
        x.append(p)
        y.append(np.average(hvol_all[ids]))
    plt.scatter(population, hvol_all)
    plt.plot(x, y, color="black")

    x = []
    y = []
    plt.subplot(2, 2, 2)
    for o in np.unique(offspring):
        ids = np.where(offspring == o)[0]
        x.append(o)
        y.append(np.average(hvol_all[ids]))
    plt.scatter(offspring, hvol_all)
    plt.plot(x, y, color="black")


    plt.subplot(2, 2, 3)
    x = []
    y = []
    for pm in np.unique(pmutation):
        ids = np.where(pmutation == pm)[0]
        x.append(pm)
        y.append(np.average(hvol_all[ids]))
    plt.scatter(pmutation, hvol_all)
    plt.plot(x, y, color="black")


    #plt.scatter(pmutation, hvol_all, s=2)
    plt.subplot(2, 2, 4)
    #plt.scatter(rmutation, hvol_all)
    plt.show()

    # Best and worst runs
    print(ids[np.argmax(np.array(hvol_all))], ids[np.argmin(np.array(hvol_all))])
