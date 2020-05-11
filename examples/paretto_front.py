import source.io_util as io


def sort_results_by_params(R, phase=""):
    if phase == "train":
        return sorted(R.items(), key=lambda item: item[1].train['system'].params)
    elif phase == "test":
        return sorted(R.items(), key=lambda item: item[1].test['system'].params)
    elif phase == "val":
        return sorted(R.items(), key=lambda item: item[1].val['system'].params)
    else:
        return sorted(R.items(), key=lambda item: item[1]['system'].params)


def sort_results_by_time(R, phase=""):
    if phase == "train":
        return sorted(R.items(), key=lambda item: item[1].train['system'].time)
    elif phase == "test":
        return sorted(R.items(), key=lambda item: item[1].test['system'].time)
    elif phase == "val":
        return sorted(R.items(), key=lambda item: item[1].val['system'].time)
    else:
        return sorted(R.items(), key=lambda item: item[1]['system'].time)


def sort_results_by_accuracy(R, phase=""):
    if phase == "train":
        return sorted(R.items(), key=lambda item: item[1].train['system'].accuracy)
    elif phase == "test":
        return sorted(R.items(), key=lambda item: item[1].test['system'].accuracy)
    elif phase == "val":
        return sorted(R.items(), key=lambda item: item[1].val['system'].accuracy)
    else:
        return sorted(R.items(), key=lambda item: item[1]['system'].accuracy)


def get_front_params_accuracy(R, phase=""):
    sorted_R = sort_results_by_params(R, phase)
    paretto_models = []
    max_acc = 0
    for key, value in sorted_R:
        if phase == "train":
            accuracy = value.train['system'].accuracy
        elif phase == "test":
            accuracy = value.test['system'].accuracy
        elif phase == "val":
            accuracy = value.val['system'].accuracy
        else: accuracy = value['system'].accuracy
        if accuracy > max_acc:
            paretto_models.append((key, value))
            max_acc = accuracy
    return dict(paretto_models)


def get_front_time_accuracy(R, phase=""):
    sorted_R = sort_results_by_time(R, phase)
    paretto_models = []
    max_acc = 0
    for key, value in sorted_R:
        if phase == "train":
            accuracy = value.train['system'].accuracy
        elif phase == "test":
            accuracy = value.test['system'].accuracy
        elif phase == "val":
            accuracy = value.val['system'].accuracy
        else: accuracy = value['system'].accuracy
        if accuracy > max_acc:
            paretto_models.append((key, value))
            max_acc = accuracy
    return dict(paretto_models)


def get_front_accuracy_time(R):
    sorted_R = sorted(R.items(), key=lambda item: item[1].accuracy)
    paretto_models = []
    max = 0
    for key, value in sorted_R:
        if value.time > max:
            paretto_models.append((key, value))
            max = value.time
    return dict(paretto_models)


def get_front_accuracy_params(R):
    sorted_R = sorted(R.items(), key=lambda item: item[1].accuracy if "system" not in item[0] else item[1]['system'].accuracy)
    paretto_models = []
    max = 0
    for key, value in sorted_R:
        params = value.params if "system" not in key else value['system'].params
        if params > max:
            paretto_models.append((key, value))
            max = params
    return dict(paretto_models)


def get_speedup_id(f0, f1, e, phase = None):
    """
    SUMMARY :Given a resource, compute the speedup of one front vs the other.
            Speedup = same accuracy less time
    :param f0: Front of classifiers/models 1
    :param f1: Front of classifiers/models 2
    :param e: Epsilon = ammount of resource (time,params)
    :return: Speedup of the front1 vs front0 given e
    """
    if phase == "test":
        f0 = [(k, f.test) for k, f in f0]
        f1 = [(k, f.test) for k, f in f1]

    acc = 0
    for k, v in f0:
        if v['system'].time < e:
            acc = v['system'].accuracy

    id = ""
    for k, v in f1:
        if v['system'].time < e:
            id = k
        if v['system'].accuracy > acc:
            break
    return id


def get_speedup_front(f0, f1, e, phase = None):
    """
    SUMMARY :Given a resource, compute the speedup of one front vs the other.
            Speedup = same accuracy less time
    :param f0: Front of classifiers/models 1
    :param f1: Front of classifiers/models 2
    :param e: Epsilon = ammount of resource (time,params)
    :return: Speedup of the front1 vs front0 given e
    """
    if phase == "test":
        f0 = [(k, f.test) for k, f in f0]
        f1 = [(k, f.test) for k, f in f1]

    acc = 0
    time = 0
    for k, v in f0:
        if v['system'].time < e:
            acc = v['system'].accuracy
            time = v['system'].time

    speedup = 0
    for k, v in f1:
        if v['system'].time < e:
            speedup = time/v['system'].time
        if v['system'].accuracy >= acc:
            break
    return speedup


def get_decrease_params_front(f0, f1, e, phase = None):
    """
    SUMMARY :Given a resource, compute the speedup of one front vs the other.
            Speedup = same accuracy less time
    :param f0: Front of classifiers/models 1
    :param f1: Front of classifiers/models 2
    :param e: Epsilon = ammount of resource (time,params)
    :return: Speedup of the front1 vs front0 given e
    """
    if phase == "test":
        f0 = [(k, f.test) for k, f in f0]
        f1 = [(k, f.test) for k, f in f1]

    acc = 0
    params = 0
    for k, v in f0:
        if v['system'].params < e:
            acc = v['system'].accuracy
            params = v['system'].params

    decrease = 1
    for k, v in f1:
        if v['system'].params < e:
            decrease = v['system'].params/params
        if v['system'].accuracy > acc: break
    return decrease


def get_increment_front(f0, f1, t, phase=None):
    """
    SUMMARY :Given a resource, compute the increment of one front vs the other.
            Speedup = more accuracy less time
    :param f0: Front of classifiers/models 1
    :param f1: Front of classifiers/models 2
    :param e: Epsilon = ammount of resource (time,params)
    :return: Speedup of the front1 vs front0 given e
    """

    if phase == "test":
        f0 = [(k,f.test) for k,f in f0]
        f1 = [(k,f.test) for k,f in f1]

    acc = 0
    for k, v in f0:
        if v['system'].time < t:
            acc = v['system'].accuracy

    increment = -1
    for k, v in f1:
        i = v['system'].accuracy - acc
        if v['system'].time < t and i > increment:
            increment = i
    return increment


if __name__ == "__main__":
    R = io.read_pickle("./Results/R_2_3_part0")
    front_params = get_front_params_accuracy(R)
    print('\n'.join(front_params.keys()))
