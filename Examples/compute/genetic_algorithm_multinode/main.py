import Examples.compute.bagging_boosting_of_chains_GA.main as main
import Examples.metadata_manager_results as manager_results
import Source.genetic_algorithm.selection as selection
from Source.genetic_algorithm.fitting_functions import f1_time_param_penalization as fit
import Source.io_util as io
from mpi4py import MPI
import sys, random, os, time


def save_results(R_dict, individuals_fitness_per_generation):
    meta_data_file = os.path.join(os.environ['FCM'],
                                  'Examples',
                                  'compute',
                                  'genetic_algorithm_multinode',
                                  'results',
                                  'metadata.json')

    id = str(random.randint(0, 1e16))
    results_loc = os.path.join('Examples/compute/genetic_algorithm_multinode/results', main.args.dataset, id)
    comments = main.args.comment
    meta_data_result = manager_results.metadata_template(id, main.args.dataset, results_loc, comments)

    params = main.args.__dict__
    # Save dictionary of ensemble results
    manager_results.save_results(meta_data_file, meta_data_result, params, R_dict)
    # Additionally save ensemble results per generation (list)
    io.save_pickle(os.path.join(os.environ['FCM'], results_loc, 'individuals_fitness_per_generation.pkl'), individuals_fitness_per_generation)


def multinode_earn(comm):
    r = comm.Get_rank()
    s = comm.Get_size()

    P = P_fit = R_dict = None
    individuals_fitness_per_generation = []

    if r == 0:
        P = main.generate_initial_population()
        R = main.evaluate_population(P, phases=["val", "test"])
        P_fit = fit(R, main.args.a)
        R_dict = {}  # Evaluation results

        for i, p in enumerate(P):
            R_dict[p.get_sysid()] = R[i]

    for epoch in range(main.args.iterations):

        # 1) Send population and fitness to all nodes
        if r == 0:
            start = time.time()
            manager_data = {'individuals': P, 'fitness': P_fit, 'O': main.args.offspring // s}
        else:
            manager_data = None
        manager_data = comm.bcast(manager_data, root=0)

        # 2) Nodes receive data
        P = manager_data['individuals']
        P_fit = manager_data['fitness']
        O = manager_data['O']

        # 3) Every Node: Generate and evaluate offspring and fitness
        P_offspring_worker = main.generate_offspring(P, P_fit, O)
        R_offspring_worker = main.evaluate_population(P_offspring_worker, phases=["val", "test"])
        # fit_offspring_worker = fit_fun.f1_time_param_penalization(R_offspring_worker, main.args.a)
        # worker_data = {'offspring': P_offspring_worker, 'fitness': fit_offspring_worker, 'R': R_offspring_worker}
        worker_data = {'offspring': P_offspring_worker, 'R': R_offspring_worker}

        # 4) Send work back to manager node (rank=0)
        workers_data = comm.gather(worker_data, root=0)

        # 5) Manager Node: Receive work and select best individuals for next generations
        if r == 0:
            P_offspring = []
            R_offspring = []
            # fit_offspring = []
            for work in workers_data:
                P_offspring = P_offspring + work['offspring']
                R_offspring = R_offspring + work['R']
                # fit_offspring = fit_offspring + work['fitness']

            P_generation = P + P_offspring
            R_generation = R + R_offspring
            fit_generation = fit(R_generation, main.args.a)

            if main.args.selection == "nfit":
                best = selection.most_fit_selection(fit_generation, main.args.population)
            else:
                best = selection.roulette_selection(fit_generation, main.args.population)

            P = [P_generation[i] for i in best]
            R = [R_generation[i] for i in best]
            P_fit = [fit_generation[i] for i in best]

            # Save generation individuals
            for i, p in enumerate(P):
                R_dict[p.get_sysid()] = R[i]

            # Save which individuals alive every iteration
            ids = [p.get_sysid() for p in P]
            individuals_fitness_per_generation += [(ids, P_fit)]

            print("Iteration %d" % epoch)
            print("TIME: Seconds per generation: %f " % (time.time() - start))

    if r == 0:
        save_results(R_dict, individuals_fitness_per_generation)


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    main.args = main.argument_parse(sys.argv[1:])

    os.environ['TMP'] = 'Definitions/Classifiers/tmp/' + main.args.dataset[12:-15]
    if not os.path.exists(os.path.join(os.environ['FCM'], os.environ['TMP'])):
        os.makedirs(os.path.join(os.environ['FCM'], os.environ['TMP']))
    random.seed()

    multinode_earn(comm)
