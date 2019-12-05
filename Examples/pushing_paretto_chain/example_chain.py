import numpy as np
import Source.FastComposedModels_pb2 as fcm
import Source.system_evaluator as eval
import Source.system_builder as sb
import Source.io_util as io
import Source.make_util as make
import Examples.paretto_front as paretto
import os

def update_dataset(model_file, th, train_path, test_path):
    model = io.read_pickle(model_file)
    # Test
    dataset_test = model['test']
    dataset_test['th'] = th
    io.save_pickle(test_path, dataset_test)
    # Train
    dataset_train = model['train']
    dataset_train['th'] = th
    io.save_pickle(train_path, dataset_train)

if __name__ == "__main__":

    path = os.environ['PYTHONPATH']

    front = io.read_pickle("./results/front1")
    front_sorted = paretto.sort_results_by_accuracy(front)
    front_sorted_id = [item[0] for item in front_sorted]

    R = {}

    for mi, m in enumerate(front_sorted_id):
        sys = sb.SystemBuilder(verbose=False)
        last_small = ""

        if "system" not in m:
            print("Beginner Small:" + m)
            a = make.make_classifier("beginning_small", m, "trigger_last_small")
            sys.replace("beginning_small", a)
            last_small = m
        else:
            print("Begginner Small:" + front[m]['small_model'])
            # Small
            a = make.make_classifier("beginning_small", front[m]['small_model'], "trigger_beginning_small")
            sys.replace("beginning_small", a)

            # Trigger data
            train_path = path + '/Data/train_trigger_beginning_small_threshold.pkl'
            test_path = path + '/Data/test_trigger_beginning_small_threshold.pkl'
            source = make.make_source(train_path, test_path, fcm.Data.Source.NUMPY)
            data = make.make_data("trigger_beginning_small_data", int(5e4), int(1e4), source=source)
            sys.replace("trigger_beginning_small_data", data)

            update_dataset(front[m]['small_model'], front[m]['th'], train_path, test_path)

            # Trigger
            trigger_a = make.make_trigger("trigger_beginning_small",
                                          make.make_empty_classifier(data_id="trigger_beginning_small_data"),
                                          ["last_small"], model="probability")
            sys.replace("trigger_beginning_small", trigger_a)

            # Big
            b = make.make_classifier("last_small", front[m]['big_model'], "trigger_last_small")
            sys.replace("last_small", b)
            last_small = front[m]['big_model']

            print("Last Small:", front[m]['big_model'])

        for th in np.arange(0.1, 1, 0.1):

            print("\t Threshold:" + str(th))

            # Trigger data
            train_path = path + '/Data/train_trigger_last_small_threshold.pkl'
            test_path = path + '/Data/test_trigger_last_small_threshold.pkl'
            source = make.make_source(train_path, test_path, fcm.Data.Source.NUMPY)
            data = make.make_data("trigger_last_small_data", int(5e4), int(1e4), source=source)
            sys.replace("trigger_last_small_data", data)

            update_dataset(last_small, th, train_path, test_path)

            trigger_b = make.make_trigger("trigger_last_small",
                                          make.make_empty_classifier("", "trigger_last_small_data"),
                                          ["beginning_big"], model="probability")
            sys.replace("trigger_last_small", trigger_b)

            for m_ in front_sorted_id[mi+1:]:

                if "system" in m_:

                    print("\t\t Begginner Big:" + front[m_]['small_model'])
                    # Small
                    a = make.make_classifier("beginning_big", front[m_]['small_model'], "trigger_beginning_big")
                    sys.replace("beginning_big", a)

                    # Trigger data
                    train_path = path + '/Data/train_trigger_beginning_big_threshold.pkl'
                    test_path = path + '/Data/test_trigger_beginning_big_threshold.pkl'
                    source = make.make_source(train_path, test_path, fcm.Data.Source.NUMPY)
                    data = make.make_data("trigger_beginning_big_data", int(5e4), int(1e4), source=source)
                    sys.replace("trigger_beginning_big_data", data)

                    update_dataset(front[m_]['small_model'], front[m_]['th'], train_path, test_path)

                    # Trigger
                    trigger_b = make.make_trigger("trigger_beginning_big",
                                                  make.make_empty_classifier(data_id="trigger_beginning_big_data"),
                                                  ["last_big"], model="probability")
                    sys.replace("trigger_beginning_big", trigger_b)

                    # Big
                    b = make.make_classifier("last_big", front[m_]['big_model'])
                    sys.replace("last_big", b)

                    print("\t\t Last Big:", front[m_]['big_model'])

                else:
                    print(" \t\t Last big: " + m_)
                    c = make.make_classifier("beginning_big", m_)
                    sys.replace("beginning_big", c)

                result = eval.evaluate(sys, "beginning_small")
                R[m+m_] = result.test

    import Examples.plot as myplt
    myplt.plot_accuracy_time(R)
    io.save_pickle("./results/example_chain/R_all", R)





