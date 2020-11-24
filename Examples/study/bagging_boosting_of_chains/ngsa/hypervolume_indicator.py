from Source.genetic_algorithm.fitting_functions import normalize_error_time_params as f
from Source.genetic_algorithm.fitting_functions import make_limits_dict, update_limit_dict
from Source.genetic_algorithm.moo import compute_hvolume
from Examples import metadata_manager_results as results_manager
import matplotlib.pyplot as plt
import Source.io_util as io
import matplotlib.cm as cm
import numpy as np
import os


def hvolume_evolution(R: dict, individuals_gen: list, a: np.ndarray, phase="test", start=-1) -> np.ndarray:

    """

    :param R: Dict storing ensembles evaluated (key = ensemble id, value = Source.system_evaluator.Results object)
    :param individuals_gen: [ (ids, fit), (ids, fit) ... ]. Each element stores the ids and fitness value of the ensembles present in the generation
    :param a: Array of integers. Indicates which objectives take into account to compute the hypervolume (1) or not (0)
    :param phase: test or val
    :param start: From which iteration of EARN compute the hypervolume
    :return:
    """

    assert phase in ["test", "val"]
    assert start < len(individuals_gen)
    assert 0 < a.shape[0] < 4

    # NN evaluations
    R_models = dict([(k, r) for k, r in R.items() if len(r.val if phase == "val" else r.test) < 3])
    limits = make_limits_dict()
    update_limit_dict(limits, R_models, phase)

    # Reference point
    ref = np.array([1.]*np.count_nonzero(a))  # Worst solution possible regarding NN pool of 32 models

    # Return evolution of the hypervolume
    hvolume = []
    for gen in individuals_gen[start:]:
        Rgen = [R[g] for g in gen[0]]
        obj = f(Rgen, limits, phase)
        obj = obj[:, a > 0]
        hvolume.append(compute_hvolume(obj, ref))

    return np.array(hvolume)


if __name__ == "__main__":

    ################### Plot configurations #########################
    experiment = 'bagging_boosting_of_chains_GA'
    plt.rcParams.update({'font.size': 12})
    metadataset = os.path.join(os.environ['FCM'],
                                            'Examples',
                                            'compute',
                                            experiment,
                                            'results',
                                            'metadata.json')

    param_query = {"experiment": "main_NGSA2.py",
                    "dataset": "sota_models_cifar10-32-dev_validation",
                    "iterations": 100,
                    "a": [1, 1, 1]}

    ids = results_manager.get_ids_by_fieldval(metadataset, "params", param_query)
    phase = "test"
    ##################################################################

    # 0) Parse results that don't have rm paramter
    # 1) Compute hypervolume evolution for each run
    # 2) Group hypervolume results by EARN parameters
    # 3) Print the parameter configuration that:
        # 3.1) Obtains higher hypervolume in average
        # 3.3) Obtains the higher median hypervolume
        # 3.2) Obtains the higher minimum hypervolume

    avail_params = ["population", "offspring", "iterations", "step_th", "pm", "rm", "k"]
    dictval2tuple = lambda d: tuple([d[k] for k in avail_params])
    hvolume_by_configuration = {}

    # Compute hypervolume per EARN run
    for id in ids:
        print("Processing %s" % id)
        GA_res_loc = results_manager.get_results_by_id(metadataset, id)
        GA_params = results_manager.get_fieldval_by_id(metadataset, id, "params")[0]

        individuals_gen = io.read_pickle(os.path.join(GA_res_loc, 'individuals_fitness_per_generation.pkl'))
        R = io.read_pickle(os.path.join(GA_res_loc, 'results_ensembles.pkl'))
        a = np.array(GA_params["a"])
        hv = hvolume_evolution(R, individuals_gen, a, phase)[-1]  # Only interested in the last run

        key = dictval2tuple(GA_params)
        hvs_all = hvolume_by_configuration.get(key, [])
        hvs_all.append(hv)
        hvolume_by_configuration[key] = hvs_all

    # Obtain statistics
    print(" \n *Computing statistics on %s %s set" % (param_query["dataset"], phase))
    keys = list(hvolume_by_configuration.keys())
    vals = [hvolume_by_configuration[key] for key in keys]

    # Highest hypervolume measurements on average
    max = 0
    imax = -1
    runs = 0
    for i, hvs in enumerate(vals):
        print(keys[i], hvs)
        if sum(hvs)/len(hvs) > max and len(hvs) > 1:
            max = sum(hvs)/len(hvs)
            imax = i
            runs = len(hvs)
    print()
    print("Highest average hypervolume results ", list(zip(avail_params, keys[imax])))
    print("\t\t Average hypervolume: %f" % max)
    print("\t\t Runs: %d" % runs)

    # Highest median hypervolume
    max = 0
    imax = -1
    runs = 0
    for i, hvs in enumerate(vals):
        hvs.sort()
        median = hvs[len(hvs)//2]
        if median > max and len(hvs) > 1:
            max = median
            imax = i
            runs = len(hvs)
    print(" Highest median hypervolume results ", list(zip(avail_params, keys[imax])))
    print("\t\t Average hypervolume: %f" % max)
    print("\t\t Runs: %d" % runs)

    # Highest minimum hypervolume
    max = 0
    imax = -1
    runs = 0
    for i, hvs in enumerate(vals):
        v = min(hvs)
        if v > max and len(hvs) > 1:
            max = v
            imax = i
            runs = len(hvs)
    print("Highest minimum hypervolume results ", list(zip(avail_params, keys[imax])))
    print("\t\t Average hypervolume: %f" % max)
    print("\t\t Runs: %d" % runs)


