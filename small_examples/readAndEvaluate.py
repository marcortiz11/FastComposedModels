import numpy as np
import source.system_builder as sb
import source.make_util as make
import source.io_util as io
import source.FastComposedModels_pb2 as fcm
import os
import time

if __name__ == '__main__':
    sys = sb.SystemBuilder(verbose=False)
    num_train = 50000
    num_test = 10000

    # Data
    data = make.make_data("CIFAR-10", num_train, num_test, [])
    sys.add_data(data)

    # Load classifier 1 (Error)
    name = "../definitions/Classifiers/V001_ResNet18_ref_0.prototxt"
    name = "../definitions/Classifiers/V001_ResNet20_CIFAR100.prototxt"
    # name = "/dataT/eid/GIT/tpml/ml_experiments/001interns/V001_DPN26_ref_0"
    name = "/dataT/eid/GIT/tpml/ml_experiments/001interns/V001_DPN26_ref_0.prototxt"

    model = make.make_empty_classifier()

    print("Load {} ...".format(name))
    t_start = time.time()
    io.read_message(name, model, suffix="")
    t_duration_load = time.time() - t_start

    print("Adding ...")
    t_start = time.time()
    sys.add_classifier(model)
    t_duration_add = time.time() - t_start

    # Evaluate
    print("Evaluate ...")
    t_start = time.time()
    results = sys.evaluate(model.id)
    t_duration_eval = time.time() - t_start

    print(model.id + ":")
    print("\t accuracy=", results.test['system'].accuracy)
    print("\t #parameters=", results.test['system'].parameters)
    print("\t inference time=", results.test['system'].time, "sec")
    print("\t #operations=", results.test['system'].ops)

    print("")
    print("PROTO PERFROMANCE:")
    print(" load: {:.3f} secs".format(t_duration_load))
    print("  add: {:.3f} secs".format(t_duration_add))
    print(" eval: {:.3f} secs".format(t_duration_eval))
    print("")
    print("ALL DONE")