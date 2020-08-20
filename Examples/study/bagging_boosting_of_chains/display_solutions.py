import Examples.metadata_manager_results as results_manager
import Examples.study.plot as myplt
import Source.io_util as io
import matplotlib.cm as cm
import numpy as np
import os


if __name__ == "__main__":

    # Information about the experiment
    experiment = 'bagging_boosting_of_chains_GA'
    experiment_dir = os.path.join(os.environ['FCM'], 'Examples', 'compute', experiment)
    meta_data_file = os.path.join(experiment_dir, 'results', 'metadata.json')

    ids = [1005780872640552, ]
    cmap = cm.get_cmap('jet')
    colors = cmap(np.linspace(0, 1.0, len(ids)))
    labels = ["w1=1; w2=0", "w1=0.8; w2=0.2", "w1=0.7; w2=0.3", "w1=0.6; w2=0.4"]
    phase = 'test'

    for j, id in enumerate(ids):
        chain_data_path = os.path.join(experiment_dir, results_manager.get_results_by_id(meta_data_file, str(id)),
                                       'results_ensembles.pkl')
        dataset = results_manager.get_fieldval_by_id(meta_data_file, str(id), 'dataset')[0][12:-18]

        chain = io.read_pickle(chain_data_path)
        myplt.plot_accuracy_parameters_time(chain, dataset + ' ' + phase + ' solution space', system_color=colors[j],
                                            label=labels[j], phase=phase)
    myplt.show()
