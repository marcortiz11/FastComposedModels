import Examples.metadata_manager_results as results_manager
import Examples.plot as myplt
import Examples.paretto_front as front
import Source.io_util as io
import os

if __name__ == "__main__":

    # Information about the experiment
    experiment = 'chain_genetic_algorithm_multinode'
    experiment_dir = os.path.join(os.environ['FCM'], 'Examples', 'compute', experiment)
    meta_data_file = os.path.join(experiment_dir, 'results', 'metadata.json')
    id = "2313347320959223"

    # Retrieve the results of the experiments
    # models_data_path = os.path.join(os.environ['FCM'], 'small_examples', 'models_evaluation', dataset, 'models_cpu.pkl')
    chain_data_path = os.path.join(experiment_dir, results_manager.get_results_by_id(meta_data_file, id), 'results_ensembles.pkl')
    dataset = results_manager.get_fieldval_by_id(meta_data_file, id, 'dataset')[0][12:-18]

    chain = io.read_pickle(chain_data_path)

    myplt.plot_accuracy_parameters_time(chain, dataset+' test solution space')
    myplt.show()

