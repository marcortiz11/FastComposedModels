import Source.pytorch as systorch
from Source.io_util import read_pickle
from Source.pytorch.genetic_operations import mutations
from Data.datasets import Split
import torch
import os
import unittest


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        # Merging the output predictions of a chain of 3 classifiers
        self.c1 = systorch.ClassifierMetadata(
            os.environ["FCM"]+"/Definitions/Classifiers/sota_models_cifar10-32-dev_validation/V001_DenseNet121_ref_0.pkl")
        self.c2 = systorch.ClassifierMetadata(
            os.environ["FCM"]+"/Definitions/Classifiers/sota_models_cifar10-32-dev_validation/V001_DenseNet121_ref_0.pkl")
        self.t1 = systorch.Trigger(0.0)
        self.t2 = systorch.Trigger(0.0)
        self.chain = systorch.Chain([self.c1, self.t1, self.c2, self.t2])
        self.merger = systorch.Merger([self.chain])
        self.system = systorch.System(self.chain, Split.TEST)

    def test_replace_classifier_metadata(self):
        path = os.environ["FCM"]+"/Definitions/Classifiers/sota_models_cifar10-32-dev_validation/V001_PreActResNet34_ref_0.pkl"
        mutations.replace_classifier_metadata(self.c1, path)
        self.assertEqual(True, self.c1.get_model() == path)

    def test_update_threshold(self):
        mutations.update_threshold(self.t1, 1.0)
        self.assertEqual(True, self.t1.get_threshold_value() == 1.0)

    def test_extend_chain(self):
        path = os.environ["FCM"] + "/Definitions/Classifiers/sota_models_cifar10-32-dev_validation/V001_MobileNetV2_ref_0.pkl"
        len_prev = len(self.chain.get_chained_modules())
        mutations.extend_chain(self.chain, path)
        tail_chain = self.chain.get_chained_modules()[-1]
        correct = isinstance(tail_chain, systorch.ClassifierMetadata) and tail_chain.get_model() == path
        correct = correct and (isinstance(self.chain.get_chained_modules()[-2], systorch.Trigger))
        correct = correct and len(self.chain.get_chained_modules()) == len_prev + 2   # Classifier and Trigger
        self.assertEqual(True, correct)

    def test_add_classifier_to_merger(self):
        path = os.environ["FCM"] + "/Definitions/Classifiers/sota_models_cifar10-32-dev_validation/V001_MobileNetV2_ref_0.pkl"
        len_merged = len(self.merger.get_merged_modules())
        mutations.add_classifier_to_merger(self.merger, path)
        self.assertEqual(True, len_merged + 1 == len(self.merger.get_merged_modules()))


if __name__ == '__main__':
    unittest.main()
