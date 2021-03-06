from Source.pytorch.classifier_metadata import ClassifierMetadata
from Source.pytorch.trigger import Trigger
from Source.pytorch.chain import Chain
from Source.pytorch.merger import Merger, MergeProtocol
import os


# Classifier mutations
def replace_classifier_metadata(c: ClassifierMetadata, path: str):
    if not os.path.exists(path):
        raise ValueError("Classifier's metadata with path specified does not exist")

    c.update_model(path)


# Trigger mutations
def update_threshold(c: Trigger, new_th: float):
    if new_th < 0 or new_th > 1:
        raise ValueError("Trigger's threshold ranges between 0..1")

    c.update_threshold(new_th)


# Chain mutation
def extend_chain(c: Chain, path: str, th=0.5):
    if not os.path.exists(path):
        raise ValueError("Classifier's metadata with path specified does not exist")
    if 0 > th > 1:
        raise ValueError("Trigger's threshold ranges between 0..1")

    classifier = ClassifierMetadata(path)
    trigger = Trigger(th)
    c.extend_chain([trigger, classifier])


# Merger mutations
def add_classifier_to_merger(c: Merger, path: str):
    if not os.path.exists(path):
        raise ValueError("Classifier's metadata with path specified does not exist")

    c.add_classifier(c)


def change_merging_protocol(c: Merger, protocol: MergeProtocol):
    c.update_merge_protocol(protocol)

