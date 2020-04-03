import Examples.metadata_manager_results as results_manager
import Examples.plot as myplt
import Examples.paretto_front as front
import Source.io_util as io
import os

if __name__ == "__main__":

    # Information about the experiment
    dataset = 'sota_models_cifar10-32-dev_validation'
    experiment = 'chain_genetic_algorithm'
    #experiment = 'fully_connected_chain'
    experiment_dir = os.path.join(os.environ['FCM'], 'Examples', 'Compute', experiment)
    meta_data_file = os.path.join(experiment_dir, 'results', 'metadata.json')
    id = "844200286319898"

    # Retrieve the results of the experiments
    models_data_path = os.path.join(os.environ['FCM'], 'SmallSamples', 'models_evaluation', dataset, 'models.pkl')
    chain_data_path = os.path.join(experiment_dir, results_manager.get_results_by_id(meta_data_file, id))

    chain = io.read_pickle(chain_data_path)
    models = io.read_pickle(models_data_path)

    myplt.plot_accuracy_time_old(chain)
    myplt.plot_accuracy_time_old(models)
    myplt.show()

