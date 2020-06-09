# Delete Data from triggers from not selected individuals!!!!!
import numpy as np


def spin_roulette(fit_vals, mask=None):

    """
    Action if spinning the wheel
    :param fit_vals: Fitness values of the population
    :param mask: Boolean vector. Tells which individuals consider this time
    :return: index of selected individual among population
    """

    n_individuals = len(fit_vals)

    assert n_individuals > 0, "ERROR: Fitness values empty list"
    assert min(fit_vals) > 0, "ERROR: Roulette selection only works with positive fitness values"

    if mask is not None:
        assert len(mask) == n_individuals, "ERROR: Length of mask should be length of fitness values"

    scaled_fit_vals = np.array(fit_vals)
    upper_limit = sum(scaled_fit_vals * mask if mask is not None else scaled_fit_vals)
    r = np.random.uniform(0, upper_limit)
    j = accumulation = 0

    while accumulation <= r and j < n_individuals:
        if mask is None or mask[j]:
            accumulation += scaled_fit_vals[j]
        j += 1

    return j-1


def roulette_selection(fit_vals, num_population):
    """
    Selects num_population individuals using the rank selection algorithm
    :param fit_vals: Vector of fitting values
    :param num_population: Number of individuals to select
    :return: ids of the population for the next generation
    """
    num_population = num_population if num_population <= len(fit_vals) else len(fit_vals)

    mask = [1]*len(fit_vals)  # Mask. we don't want to consider already selected individuals
    selected_ids = []

    for i in range(num_population):
        selected = spin_roulette(fit_vals, mask)  # Run the roulette wheel
        selected_ids.append(selected)
        mask[selected] = 0  # Do not consider selected for next spin

    return selected_ids


def most_fit_selection(fit_vals, num_population):

    """
    Selects the num_population most fit individuals
    :param fit_vals: Vector of fitting values
    :param num_population: Number of individuals to select
    :return: ids of the population for the next generation
    """

    num_population = num_population if num_population <= len(fit_vals) else len(fit_vals)
    selected_ids = np.argsort(fit_vals)[-num_population:]
    return selected_ids
