# Delete Data from triggers from not selected individuals!!!!!
import numpy as np


def random_rank_selection(fit_vals, num_population):
    """
    Selects num_population individuals using the rank selection algorithm
    :param fit_vals: Vector of fitting values
    :param num_population: Number of individuals to select
    :return: ids of the population for the next generation
    """
    num_population = num_population if num_population <= len(fit_vals) else len(fit_vals)

    fit_vals = fit_vals - min(fit_vals)
    limit = np.sum(fit_vals)
    sorted = np.argsort(fit_vals)
    selected_ids = []

    for i in range(num_population):
        accumulation = 0
        j = 0
        r = np.random.uniform(0, limit)
        while accumulation < r and j < len(sorted):
            accumulation += fit_vals[sorted[j]]
            j += 1
        selected_ids.append(sorted[j-1])
        limit = max(0, limit - fit_vals[sorted[j-1]])
        sorted = np.delete(sorted, j-1)

    return selected_ids

def rank_selection(fit_vals, num_population):
    """
    Selects num_population individuals using the rank selection algorithm
    :param fit_vals: Vector of fitting values
    :param num_population: Number of individuals to select
    :return: ids of the population for the next generation
    """

    num_population = num_population if num_population <= len(fit_vals) else len(fit_vals)
    sorted = np.argsort(-fit_vals)
    return sorted[:num_population]

