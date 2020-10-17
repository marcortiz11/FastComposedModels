import random
import os
import json
import Source.io_util as io
import shutil
from datetime import date


def metadata_template(id, dataset, results_loc, comments=""):

    """
    Returns a standard python dictionary of the meta_data of the computed ensembles results.
    :param dataset: Dataset of the experiment
    :param results_loc: Desired of the ensemble results (train+test+validation)
    :param id: Id of the computation
    :param comments: Additional comments on the computed ensemble results
    :return: Python dictionary
    """

    assert id != "", "Id of the computed ensembles cannot be empty"

    from time import gmtime, strftime
    meta_data = {
        "id": id,
        "date": strftime("%d-%m-%Y %H:%M:%S", gmtime()),
        "dataset": dataset,
        "results": results_loc,
        "comments": comments,
        "params": {}  # Specific to every ensemble type or ensemble experiment. Filled later
    }

    return meta_data


def save(meta_file, meta_data_results, params, *results):

    """
    Saves the results pickle and checks meta-Data information about the results is correct and updated

    :param meta_file: Storage of all the ensemble results for a particular experiment/ensemble type.
    :param meta_data_results: Meta Data information of a new result
    :param params: Params of the experiment
    :return:
    """

    assert os.path.isabs(meta_file), "ERROR Metadata Results: File meta_file should be an absolute path to the file"
    assert meta_data_results['results'] != "" and meta_data_results['results'] is not None

    if not os.path.exists(meta_file):
        with open(meta_file, 'w') as file:
            json.dump([], file)

    save_path = os.path.join(os.environ['FCM'], meta_data_results['results'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 1- Guardar R
    for r in results:
        io.save_pickle(os.path.join(save_path, r[0]), r[1])

    # 2- Obre el meta_data file i fer update
    meta_data_results['params'] = params
    with open(meta_file, 'r') as file:
        meta_data = json.load(file)

    with open(meta_file, 'w') as file:
        meta_data.append(meta_data_results)
        json.dump(meta_data, file, indent=4)


def save_results(meta_file, meta_data_results, params, R):

    """
    Saves the results pickle and checks meta-Data information about the results is correct and updated

    :param meta_file: Storage of all the ensemble results for a particular experiment/ensemble type.
    :param meta_data_results: Meta Data information of a new result
    :param params: Params of the experiment
    :return:
    """

    assert os.path.isabs(meta_file), "ERROR Metadata Results: File meta_file should be an absolute path to the file"
    assert meta_data_results['results'] != "" and meta_data_results['results'] is not None

    if not os.path.exists(meta_file):
        with open(meta_file, 'w') as file:
            json.dump([], file)

    save_path = os.path.join(os.environ['FCM'], meta_data_results['results'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 1- Guardar R
    io.save_pickle(os.path.join(save_path, "results_ensembles.pkl"), R)

    # 2- Obre el meta_data file i fer update
    meta_data_results['params'] = params
    with open(meta_file, 'r') as file:
        meta_data = json.load(file)

    with open(meta_file, 'w') as file:
        meta_data.append(meta_data_results)
        json.dump(meta_data, file, indent=4)


def save_results_and_ensembles(meta_file, meta_data_results, params, R, E):

    """
    Saves the set of ensemble classifier E, together with the evaluation R of the ensembles

    :param meta_file: Storage of all the ensemble results for a particular experiment/ensemble type.
    :param meta_data_results: Meta Data information of a new result
    :param params: Params of the experiment
    :return:
    """

    assert os.path.isabs(meta_file), "ERROR Metadata Results: File meta_file should be an absolute path to the file"
    assert meta_data_results['results'] != "" and meta_data_results['results'] is not None

    if not os.path.exists(meta_file):
        with open(meta_file, 'w') as file:
            json.dump([], file)

    save_path = os.path.join(os.environ['FCM'], meta_data_results['results'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 1- Guardar R
    io.save_pickle(os.path.join(save_path, "results_ensembles.pkl"), R)
    io.save_pickle(os.path.join(save_path, "ensembles.pkl"), E)

    # 2- Obre el meta_data file i fer update
    meta_data_results['params'] = params
    with open(meta_file, 'r') as file:
        meta_data = json.load(file)

    with open(meta_file, 'w') as file:
        meta_data.append(meta_data_results)
        json.dump(meta_data, file, indent=4)


def get_results_by_id(meta_file, id):
    """
    Returns the evaluation results of a set of ensembles with experiment id = id
    :param meta_file: Dataset metadata
    :param id: Id of experiment
    :return: File location of the results
    """
    id = str(id)
    with open(meta_file) as file:
        meta_dataset = json.load(file)
        for entry in meta_dataset:
            if entry['id'] == id:
                return os.path.join(os.environ['FCM'], entry['results'])
    return None


def get_results_by_params(meta_file, param):
    """
    Returns the evaluation results of a set of ensembles with parameter values = param
    :param meta_file: Dataset metadata
    :param param: Input parameters of the python execution
    :return: File location of the results
    """
    result_locations = []
    with open(meta_file) as file:
        meta_dataset = json.load(file)
        for entry in meta_dataset:
            param_entry = entry['params']
            valid = True
            for key in param.keys():
                if key not in param_entry or param_entry[key] != param[key]:
                    valid = False
                    break
            if valid:
                result_locations += [os.path.join(os.environ['FCM'], entry['results'])]
    return result_locations


def get_ensembles_by_id(meta_file, id):
    """
    Returns the ensembles evaluated with experiment id = id
    :param meta_file: Dataset metadata
    :param id: Id of experiment
    :return: File location of the ensembles
    """
    id = str(id)
    with open(meta_file) as file:
        meta_dataset = json.load(file)
        for entry in meta_dataset:
            if entry['id'] == id:
                return os.path.join(os.environ['FCM'], entry['results'])
    return None


def get_fieldval_by_id(meta_file, id, *fields):
    query_result = []
    id = str(id)
    with open(meta_file) as file:
        meta_dataset = json.load(file)
        for entry in meta_dataset:
            if entry['id'] == id:
                for field in fields:
                    query_result.append(entry[field])
                return query_result
    return None


def get_ids_by_fieldval(meta_file, field, val):
    """
    Returns the experiment ids that satisfy the query experiment[field] == val
    :param meta_file:
    :param field:
    :param val:
    :return: Returns all the ids
    """
    query_result = []
    with open(meta_file) as file:
        meta_dataset = json.load(file)
        for entry in meta_dataset:
            if entry[field] == val:
                query_result.append(entry['id'])
    return query_result


def delete_by_id(meta_file, ids):

    """
    Deletes entries in the experiment dataset that have a certain id
    :param meta_file:
    :param id:
    :return:
    """

    with open(meta_file) as file:
        meta_dataset = json.load(file)
        for i, entry in enumerate(meta_dataset):
            if entry['id'] in ids:
                shutil.rmtree(os.path.join(os.environ['FCM'], entry['results']))
                meta_dataset.pop(i)

    with open(meta_file, 'w') as file:
        json.dump(meta_dataset, file, indent=4)







