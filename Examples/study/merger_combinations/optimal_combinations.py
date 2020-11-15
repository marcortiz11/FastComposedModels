import Source.io_util as io
import Examples.metadata_manager_results as result_manager
import Examples.metadata_manager_results as manres
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os


class Y_Histogram:

    def __init__(self, resolution=100, range_right=1):
        # Two histogram bins
        self.y = np.zeros(resolution)
        self.count = np.zeros(resolution)
        self.max = range_right
        self.resolution = resolution

    def update(self, y, x):
        assert x.shape == y.shape, "y and x input parameter shape must be equal"
        x = x/self.max
        bin = (x*self.resolution).astype(np.uint)
        for i, b in enumerate(bin):
            self.y[b] += y[i]
            self.count[b] += 1

    def clear(self):
        self.y *= 0
        self.count *= 0

    def get_y(self):
        return np.divide(self.y, self.count)

    def get_x(self):
        return np.arange(0, self.max, self.max/self.resolution)

    def get_count(self):
        return self.count


def plot_optimal_combinations_merger(R, resolution=100, label="", color=None, linestyle=None):
    ids = R.keys()
    y = np.array([R[id].test["system"].accuracy for id in ids if len(R[id].test.keys()) > 2])
    x_all = np.array([R[id].test[key].accuracy for id in ids if len(R[id].test.keys()) > 2 for key in R[id].test.keys() if "trigger" not in key and "system" not in key])
    x_all = x_all.reshape(-1, 3)
    x_min = np.min(x_all, axis=1)
    x_max = np.max(x_all, axis=1)
    x = x_max - x_min  # Worst vs Best DNNs in ensemble
    y = y - x_max  # Ensemble accuracy vs best DNN accuracy

    # Histogram
    histo = Y_Histogram(resolution, 1)
    histo.update(y, x)

    # Figure

    plt.plot(histo.get_x(), histo.get_y(), label=label, color=color, linestyle=linestyle)


if __name__ == "__main__":

    plt.grid(True)
    plt.title("Caltech256 - Best configurations")
    plt.ylabel("Ensemble acc. - Max acc.")
    plt.xlabel("Max acc. - Min acc.")

    metadata_file = "../../Compute/merger_combinations/results/metadata.json"

    ids = [7377603373712973,
           7104162940520812,
           1943013241626030,
           9283423602893484,
           4453353588233325,
           1600106112124184]

    label = ["Bagging average",
             "Bagging voting",
             "Bagging max",
             "Boosting average",
             "Boosting voting",
             "Boosting max"]

    cmap = cm.get_cmap('jet')
    colors = cmap(np.linspace(0, 1.0, 3))
    linestyle = ["-", "-", "-", "--", "--", "--"]

    for j, id in enumerate(ids):
        result_location = os.path.join(result_manager.get_results_by_id(metadata_file, id), "results_ensembles")
        exec_params = result_manager.get_fieldval_by_id(metadata_file, id, "params")[0]
        R = io.read_pickle(result_location)
        plot_optimal_combinations_merger(R, resolution=50, label=label[j], color=colors[j%3], linestyle=linestyle[j])

    plt.legend()
    plt.show()
