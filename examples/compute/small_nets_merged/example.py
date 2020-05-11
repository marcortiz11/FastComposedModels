import numpy as np
import source.system_builder as sb
import source.make_util as make
import source.io_util as io
import source.FastComposedModels_pb2 as fcm
import source.system_evaluator as eval
import os
import matplotlib.pyplot as plt


if __name__ == "__main__":

    Classifier_Path = "../../definitions/Classifiers"
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