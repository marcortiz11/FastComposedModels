import numpy as np


def __spin_roulette(fit_vals, mask=None):

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
        selected = __spin_roulette(fit_vals, mask)  # Run the roulette wheel
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


def linear_rank_selection(fit_vals, n_survivors):
    if n_survivors < len(fit_vals):
        N = len(fit_vals)
        S = np.argsort(fit_vals)
        P = [1/N * (2 * (i/(N-1))) for i in range(N)]
        selected = np.random.choice(N, n_survivors, p=P, replace=False)
        ids = [S[i] for i in selected]
        return ids
    else:
        return list(range(len(fit_vals)))


def tournament_selection(fit_vals:np.ndarray, K: int, *fit_vals_next: np.ndarray, p=0.8) -> int:

    """
    :param fit_vals: Fitness values of individuals.s
    :param K: Number of individuals to participate in the tournment
    :param fit_vals_next: Secondary metrics for deciding best individuals in case of ties
    param: p: Probability of selecting winner
    :return: Index of the winner of the tournment
    """

    prob_distribution_selection = np.array([p * pow((1 - p), position) for position in range(K)])
    prob_distribution_selection /= sum(prob_distribution_selection)
    tournament_individuals = np.random.choice(len(fit_vals), K, replace=False)

    if len(fit_vals_next):
        fitness_individuals = np.row_stack((fit_vals, np.array(fit_vals_next)))
        fitness_individuals_tournment = (fitness_individuals[:, tournament_individuals])[::-1]
        result_tournment = np.lexsort(-1 * fitness_individuals_tournment)
    else:
        fitness_individuals_tournment = np.array(fit_vals)[tournament_individuals]
        result_tournment = np.argsort(-1 * fitness_individuals_tournment)

    position = np.random.choice(result_tournment, 1, p=prob_distribution_selection)
    winner = tournament_individuals[position[0]]

    return winner


def dominator_selection(fit_vals):
    """
    Returns the dominator set of ensembles according to the values in fit_vals
    :param fit_vals: 2-Dim list containing: Normalized accuracy [0..1],
                     normalized prediction speed (1 - norm. inference time [0..1]),
                     and normalized ensemble size (1 - norm. #params [0..1])
                     for each ensemble evaluated
    :return: The set of dominator solutions (Paretto Frontier)
    """

    assert len(fit_vals) > 0 and len(fit_vals[0]) == 3, \
        "Fitting value should have 3 components"

    dominators = set([])
    dominates = lambda f1, f2:  f1[0] >= f2[0] and f1[1] >= f2[1] and f1[2] >= f2[2]

    for i, f in enumerate(fit_vals):
        dominated = False
        for d in list(dominators):
            fd = d[1]
            if dominates(f, fd):
                dominators.remove(d)
            elif dominates(fd, f):
                dominated = True
                break
        if not dominated:
            elem = (i, f)
            dominators.add(elem)

    return [d[0] for d in dominators]


if __name__ == "__main__":
    # Testing tournment selection with multiple fitness values
    f1 = np.ones(10)
    f2 = np.arange(1, 0, -1/10)
    w = tournament_selection(f1, 3, f2)
