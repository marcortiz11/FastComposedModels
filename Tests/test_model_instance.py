import Source.system_builder as sb
import Source.protobuf.make_util as make
import Source.io_util as io
import Source.system_evaluator as eval
import Source.protobuf.FastComposedModels_pb2 as fcm
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

    # Dict with the results of the evaluations
    records = {}
    path = os.environ['PYTHONPATH']
    train_path = path+'/Data/train_trigger_threshold.pkl'
    test_path = path+'/Data/test_trigger_threshold.pkl'
    small_cfile = "../Definitions/Classifiers/DenseNet121_cifar10"
    big_cfile = "../Definitions/Classifiers/DenseNet201_cifar10"
    th = 0.9

    sys = sb.SystemBuilder(verbose=False)

    bigClassifier = make.make_classifier("big", big_cfile)
    sys.add_classifier(bigClassifier)

    source = make.make_source(train_path, test_path, fcm.Data.Source.NUMPY)
    data = make.make_data("trigger_data", int(5e4), int(1e4), source=source)
    sys.add_data(data)
    update_dataset(small_cfile, th, train_path, test_path)

    trigger = make.make_trigger("trigger", make.make_empty_classifier(data_id="trigger_data"),
                                ["big"], model="probability")
    sys.add_trigger(trigger)

    smallClassifier = make.make_classifier("small", small_cfile, "trigger")
    sys.add_classifier(smallClassifier)

    r = eval.evaluate(sys, "small")
    eval.pretty_print(r)
    print(r.test['system'].instance_model)

    time = 0
    time_small = io.read_pickle(small_cfile)['metrics']['time']/128
    time_big = io.read_pickle(big_cfile)['metrics']['time']/128
    for id, model in r.test['system'].instance_model.items():
        time += time_small if len(model) == 1 else time_small + time_big

    print(time, r.test['system'].time)

