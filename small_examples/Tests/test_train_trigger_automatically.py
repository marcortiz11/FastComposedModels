import source.system_builder as sb
import source.make_util as make
import source.io_util as io
import source.system_evaluator as eval
import source.FastComposedModels_pb2 as fcm
import os


def update_dataset(model_file, th, train_path, test_path, val_path):
    model = io.read_pickle(model_file)
    # Test
    dataset_test = model['test']
    dataset_test['th'] = th
    io.save_pickle(test_path, dataset_test)
    # Train
    dataset_train = model['train']
    dataset_train['th'] = th
    io.save_pickle(train_path, dataset_train)
    # Validation
    dataset_val = model['val']
    dataset_val['th'] = th
    io.save_pickle(val_path, dataset_val)


if __name__ == "__main__":

    # Dict with the results of the evaluations
    records = {}
    path = os.environ['FCM']
    train_path = path+'/data/train_trigger_threshold.pkl'
    test_path = path+'/data/test_trigger_threshold.pkl'
    val_path = path+'/data/val_trigger_threshold.pkl'
    small_cfile = "../definitions/Classifiers/front45_models_validation/V001_DenseNet_s1_39"
    big_cfile = "../definitions/Classifiers/front45_models_validation/V001_DenseNet_s1_66"

    sys = sb.SystemBuilder(verbose=False)

    bigClassifier = make.make_classifier("big", big_cfile)
    sys.add_classifier(bigClassifier)

    source = make.make_source(train_path, test_path, fcm.Data.Source.NUMPY, val_path)
    data = make.make_data("trigger_data", int(5e4), int(1e4), source=source)
    sys.add_data(data)
    update_dataset(small_cfile, 0.6, train_path, test_path, val_path)

    trigger = make.make_trigger("trigger", make.make_empty_classifier(data_id="trigger_data"),
                                ["big"], model="probability")
    sys.add_trigger(trigger)

    smallClassifier = make.make_classifier("small", small_cfile, "trigger")
    sys.add_classifier(smallClassifier)

    r = eval.evaluate(sys, "small", phases=['test', 'val'], check_classifiers=True)
    import source.system_evaluator_utils as eval_utils
    eval_utils.pretty_print(r)


