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
    if mask is not None:
        assert len(mask) == n_individuals, "ERROR: Length of mask should be length of fitness values"

    fit_vals = np.array(fit_vals)
    scaled_fit_vals = fit_vals-min(fit_vals)
    upper_limit = sum(scaled_fit_vals * mask if mask is not None else scaled_fit_vals)
    r = np.random.uniform(0, upper_limit)
    j = accumulation = 0

    while accumulation < r and j < n_individuals:
        if mask is None or mask[j]:
            accumulation += scaled_fit_vals[j]
        j += 1

    if j == len(fit_vals):
        j -= 1

    return j


def roulette_selection(fit_vals, num_population):
    """
    Selects num_population individuals using the rank selection algorithm
    :param fit_vals: Vector of fitting values
    :param num_population: Number of individuals to select
    :return: ids of the population for the next generation
    """
    num_population = num_population if num_population <= len(fit_vals) else len(fit_vals)

    consider = [1]*len(fit_vals)  # Mask. we don't want to consider already selected individuals
    selected_ids = []

    for i in range(num_population):
        selected = spin_roulette(fit_vals, consider)  # Run the roulette wheel
        selected_ids.append(selected)
        consider[selected] = 0  # Do not consider selected for next spin

    return selected_ids


