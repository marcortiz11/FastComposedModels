import numpy as np
import Source.system_builder as sb
import Source.protobuf.make_util as make
import Source.io_util as io
import Source.protobuf.FastComposedModels_pb2 as fcm
import Source.system_evaluator as eval
import os
import matplotlib.pyplot as plt


if __name__ == "__main__":

    Classifier_Path = "../../Definitions/Classifiers"
    models = [f for f in os.listdir(Classifier_Path) if ".pkl" in f]

    # Creating system
    sys = sb.SystemBuilder(verbose=False)

    classifiers_ids = []
    for m in models:
        print(m)
        file = Classifier_Path + '/' + m
        model = make.make_classifier(m, file)
        sys.add_classifier(model)
        classifiers_ids.append(model.id)

    merger = make.make_merger("AVG_MERGER", classifiers_ids, merge_type=fcm.Merger.VOTING)
    sys.add_merger(merger)

    results = eval.evaluate(sys, merger.id)
    print(eval.pretty_print(results))