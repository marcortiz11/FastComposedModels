import random
import os
import json
import Source.io_util as io
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

    today = date.today()
    meta_data = {
        "id": id,
        "date": today.strftime("%d/%m/%Y"),
        "dataset": dataset,
        "results": results_loc,
        "comments": comments,
        "params": {}  # Specific to every ensemble type or ensemble experiment. Filled later
    }

    return meta_data


def save_results(meta_file, meta_data_results, params, R):

    """
    Saves the results pickle and checks meta-data information about the results is correct and updated

    :param meta_file: Storage of all the ensemble results for a particular experiment/ensemble type.
    :param meta_data_results: Meta data information of a new result
    :param params: Params of the experiment
    :return:
    """

    assert os.path.isabs(meta_file), "ERROR Metadata Results: File meta_file should be an absolute path to the file"
    assert meta_data_results['results'] != "" and meta_data_results['results'] is not None

    if not os.path.exists(meta_file):
        with open(meta_file, 'w') as file:
            json.dump([], file)

    save_path = meta_data_results['results']
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 1- Guardar R
    io.save_pickle(os.path.join(save_path, "ensembles.pkl"), R)

    # 2- Obre el meta_data file i fer update
    meta_data_results['params'] = params
    with open(meta_file, 'r') as file:
        meta_data = json.load(file)

    with open(meta_file, 'w') as file:
        meta_data.append(meta_data_results)
        json.dump(meta_data, file, indent=4)


def get_results_by_id(meta_file, id):
    with open(meta_file) as file:
        meta_dataset = json.load(file)
        for entry in meta_dataset:
            if entry['id'] == id:
                return os.path.join(entry['results'], 'ensembles.pkl')
    return None


def get_by_id(meta_file, id, *fields):
    query_result = []
    with open(meta_file) as file:
        meta_dataset = json.load(file)
        for entry in meta_dataset:
            if entry['id'] == id:
                for field in fields:
                    query_result.append(entry[field])
                return query_result
    return None








