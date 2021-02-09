from  Source.pytorch.merger import MergeProtocol, Merger
from Source.pytorch.classifier_metadata import ClassifierMetadata, Split
from Source.pytorch.system import System
from Source.pytorch.chain import Chain
from Source.pytorch.trigger import Trigger
import Source.make_util as make
from Source.system_builder_serializable import SystemBuilder
from Source.system_evaluator import evaluate
import os
from time import perf_counter


def protobuf_ensemble_1() -> SystemBuilder:
    classifier_metadata_path = "C:/Users/bscuser/Projects/FastComposedModels/Definitions/Classifiers/sota_models_caltech256-32-dev_validation"
    files = [os.path.join(classifier_metadata_path, f) for f in os.listdir(classifier_metadata_path)
             if os.path.isfile(os.path.join(classifier_metadata_path, f))]
    ensemble = SystemBuilder(False)
    classifiers = [make.make_classifier(os.path.basename(f), f) for f in files]
    merger = make.make_merger("Merger", [c.id for c in classifiers], merge_type=2)
    for c in classifiers:
        ensemble.add_classifier(c)
    ensemble.add_merger(merger)
    ensemble.set_start("Merger")
    return ensemble


def pytorch_ensemble_1() -> System:
    classifier_metadata_path = "C:/Users/bscuser/Projects/FastComposedModels/Definitions/Classifiers/sota_models_caltech256-32-dev_validation"
    files = [os.path.join(classifier_metadata_path, f) for f in os.listdir(classifier_metadata_path)
             if os.path.isfile(os.path.join(classifier_metadata_path, f))]
    classifiers = [ClassifierMetadata(f) for f in files]
    merger = Merger(classifiers, MergeProtocol.AVERAGE)
    ensemble = System(merger)
    return ensemble


def pytorch_ensemble_2() -> System:
    classifier_metadata_path = "C:/Users/bscuser/Projects/FastComposedModels/Definitions/Classifiers/sota_models_caltech256-32-dev_validation"
    files = [os.path.join(classifier_metadata_path, f) for f in os.listdir(classifier_metadata_path)
             if os.path.isfile(os.path.join(classifier_metadata_path, f))]
    classifiers = [ClassifierMetadata(f) for f in files]
    triggers = [Trigger(0.9) for i in range(len(classifiers)-1)[::-1]]
    chain = Chain([item for sublist in zip(classifiers, triggers) for item in sublist])
    ensemble = System(chain)
    return ensemble


if __name__ == "__main__":

    # Measure how fast builds
    ts = perf_counter()
    ensemble = pytorch_ensemble_1()
    print(f"PyTorch merge ensemble creation time: {perf_counter()-ts: 0.4f}s")
    ts = perf_counter()
    ensemble_pb = protobuf_ensemble_1()
    print(f"Protobuf merge ensemble creation time: {perf_counter()-ts: 0.4f}s")
    print("--"*10)

    # Measure evaluation time on CPU
    ts = perf_counter()
    r = evaluate(ensemble, phases=[Split.TRAIN])
    print(f"PyTorch merge ensemble evaluation time: {perf_counter() - ts: 0.4f}s")
    ts = perf_counter()
    r_pb = evaluate(ensemble_pb, phases=[Split.TRAIN])
    print(f"Protobuf merge ensemble evaluation time: {perf_counter()-ts: 0.4f}s")
    print("--" * 10)

    print(r.test["system"].accuracy, r_pb.test["system"].accuracy)

    # Measure size of the ensembles
    import sys
    print(f"PyTorch merge ensemble object size: {sys.getsizeof(ensemble)/1e6:0.2f} MBytes")
    print(f"Protobuf merge ensemble object size: {sys.getsizeof(ensemble_pb)/1e6:0.2f} MBytes")
    print("==" * 10)

    # Measure how fast builds
    ts = perf_counter()
    ensemble = pytorch_ensemble_2()
    print(f"PyTorch chain ensemble creation time: {perf_counter() - ts: 0.4f}s")
    ts = perf_counter()

    # Measure evaluation time on CPU
    ts = perf_counter()
    r = evaluate(ensemble, phases=[Split.TEST])
    print(f"PyTorch chain ensemble evaluation time: {perf_counter() - ts: 0.4f}s")
