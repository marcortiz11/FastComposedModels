import numpy as np
import Source.protobuf.make_util as make_util
import Source.io_util as io_util
import Source.protobuf.FastComposedModels_pb2 as fcm



if __name__ == '__main__':
    # Load RAW(!) classifier
    # OLD AND NEW.
    name = "/dataT/eid/GIT/tpml/ml_experiments/001interns/demo_classifier"
    # name = "/dataT/eid/GIT/tpml/ml_experiments/001interns/demo_classifier_v001"

    msg = fcm.ClassifierRawData()
    io_util.read_message(name, msg)
    print(msg)


