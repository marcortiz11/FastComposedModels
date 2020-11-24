import Examples.metadata_manager_results as results_manager
import Source.io_util as io
import numpy as np
import os


def improvements_err_speedup_size(obj: np.ndarray, ref: np.ndarray, i_obj=0) -> np.ndarray:

    assert obj.shape[1] > i_obj and ref.shape[0] > i_obj

    valid = obj[:, i_obj] < ref[i_obj]
    obj = obj[valid]
    improvements = np.ones((obj.shape[1], obj.shape[1]))
    improvements[:, 0] *= 0

    if len(obj) > 0:
        for i in range(obj.shape[1]):
            extreme = np.argmin(obj[:, i])
            if obj[extreme, i] < ref[i]:
                improvements[i, :] = obj[extreme]
            else:
                improvements[i, :] = ref.copy()

        improvements[:, 0] = ref[0]-improvements[:, 0]
        improvements[:, 1] = ref[1]/improvements[:, 1]
        improvements[:, 2] /= ref[2]

    return improvements


def improvements_all(obj: np.ndarray, ref: np.ndarray) -> np.ndarray:

    valid = np.all(np.less(obj, ref), axis=1)
    obj = obj[valid]
    improvements = np.ones((obj.shape[1], obj.shape[1]))
    improvements[:, 0] *= 0

    if len(obj) > 0:
        for i in range(obj.shape[1]):
            extreme = np.argmin(obj[:, i])
            improvements[i, :] = obj[extreme]

        improvements[:, 0] = ref[0]-improvements[:, 0]
        improvements[:, 1] = ref[1]/improvements[:, 1]
        improvements[:, 2] /= ref[2]

    return improvements


def get_most_accurate_nn_result(R: dict, phase="val"):
    assert phase == "val" or "test"

    max = 0
    r = None
    if phase == "val":
        for k, v in R.items():
            if len(v.val) < 3 and v.val["system"].accuracy > max:
                max = v.val["system"].accuracy
                r = v.val
    else:
        for k, v in R.items():
            if len(v.test) < 3 and v.test["system"].accuracy > max:
                max = v.test["system"].accuracy
                r = v.test

    return r


def results_to_numpy(R: dict, phase="val") -> np.ndarray:
    assert phase == "val" or "test"

    if phase == "val":
        obj = np.array([[1-result[1].val["system"].accuracy,
                        result[1].val["system"].time,
                        result[1].val["system"].params] for result in R.items()])
    else:
        obj = np.array([[1-result[1].test["system"].accuracy,
                        result[1].test["system"].time,
                        result[1].test["system"].params] for result in R.items()])
    return obj


if __name__ == "__main__":

    metadata_file = os.path.join("../../../compute/bagging_boosting_of_chains_GA/results/metadata.json")
    phase = "test"
    params_match = {
            "dataset": "sota_models_svhn-32-dev_validation",
            "population": 500,
            "offspring": 200,
            "iterations": 50,
            "step_th": 0.1,
            "pm": 0.2,
            "rm": 0.8,
            "k": 1,
            "a": [
                1,
                1,
                1
            ]
    }

    ids = results_manager.get_ids_by_fieldval(metadata_file, "params", params_match)
    improvements = np.zeros((3, 3))
    ref = None

    for id in ids:
        R_path = results_manager.get_results_by_id(metadata_file, id)
        individuals_fitness_generation = io.read_pickle(os.path.join(R_path, 'individuals_fitness_per_generation.pkl'))
        R_dict = io.read_pickle(os.path.join(R_path, 'results_ensembles.pkl'))

        # Last generation of ensembles
        last_generation = individuals_fitness_generation[-1][0]
        R_dict_last = {ensemble_id: R_dict[ensemble_id] for ensemble_id in last_generation}

        # Most accurate NN as reference point
        if ref is None:
            r_ref = get_most_accurate_nn_result(R_dict, phase)
            ref = np.array([1-r_ref["system"].accuracy, r_ref["system"].time, r_ref["system"].params])

        # Evaluation results to numpy array
        obj = results_to_numpy(R_dict_last, phase)

        # Get improvements from reference point
        improvements += improvements_err_speedup_size(obj, ref)

    print()
    print("Average ensemble improvements on %s over %d runs" % (params_match["dataset"][12:], len(ids)))
    print(improvements/len(ids))
