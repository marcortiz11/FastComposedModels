import unittest
import torch
from Source.pytorch.classifier_metadata import ClassifierMetadata, Split
from Source.pytorch.trigger import Trigger
from Source.pytorch.chain import Chain
from Source.pytorch.system import System
from Source.io_util import read_pickle


class TestChain(unittest.TestCase):

    def setUp(self):
        self.c1 = ClassifierMetadata(
            "C:/Users/bscuser/Projects/FastComposedModels/Definitions/Classifiers/sota_models_cifar10-32-dev_validation/V001_DenseNet121_ref_0.pkl")
        self.c2 = ClassifierMetadata(
            "C:/Users/bscuser/Projects/FastComposedModels/Definitions/Classifiers/sota_models_cifar10-32-dev_validation/V001_DenseNet121_ref_0.pkl")
        self.c3 = ClassifierMetadata(
            "C:/Users/bscuser/Projects/FastComposedModels/Definitions/Classifiers/sota_models_cifar10-32-dev_validation/V001_MobileNetV2_ref_0.pkl")
        self.t1 = Trigger(0.0)
        self.t2 = Trigger(0.0)
        self.chain = Chain([self.c1, self.t1, self.c2, self.t2, self.c3])
        self.ensemble = System(self.chain, Split.TEST)

    def test_predictions_threshold_0(self):

        self.t1.update_threshold(0.0)
        self.t2.update_threshold(0.0)

        # Dataset info
        metadata = read_pickle("C:/Users/bscuser/Projects/FastComposedModels/Definitions/Classifiers/sota_models_cifar10-32-dev_validation/V001_DenseNet121_ref_0.pkl")
        # time = metadata["test"]["gt"]
        ids = metadata["test"]["id"]
        input = torch.range(0, len(ids)-1).long()
        predictions = self.ensemble(input)
        predictions_classifier_c3 = self.c3(input)
        equal = torch.equal(predictions, predictions_classifier_c3)
        self.assertEqual(True, equal)

    def test_predictions_threshold_1(self):

        self.t1.update_threshold(1.1)
        self.t2.update_threshold(1.1)

        # Dataset info
        metadata = read_pickle("C:/Users/bscuser/Projects/FastComposedModels/Definitions/Classifiers/sota_models_cifar10-32-dev_validation/V001_DenseNet121_ref_0.pkl")
        # time = metadata["test"]["gt"]
        ids = metadata["test"]["id"]
        input = torch.range(0, len(ids)-1).long()
        predictions = self.ensemble(input)
        predictions_classifier_c1 = self.c1(input)
        equal = torch.equal(predictions, predictions_classifier_c1)
        self.assertEqual(True, equal)

    def test_parameter_count(self):
        metadata1 = read_pickle(
            "C:/Users/bscuser/Projects/FastComposedModels/Definitions/Classifiers/sota_models_cifar10-32-dev_validation/V001_DenseNet121_ref_0.pkl")
        metadata2 = read_pickle(
            "C:/Users/bscuser/Projects/FastComposedModels/Definitions/Classifiers/sota_models_cifar10-32-dev_validation/V001_DenseNet121_ref_0.pkl")
        metadata3 = read_pickle(
            "C:/Users/bscuser/Projects/FastComposedModels/Definitions/Classifiers/sota_models_cifar10-32-dev_validation/V001_MobileNetV2_ref_0.pkl")

        true_count = metadata1["metrics"]["params"] + \
                     metadata2["metrics"]["params"] + \
                     metadata3["metrics"]["params"]

        self.assertEqual(self.ensemble.get_num_parameters(), true_count)

    def test_time_threshold_0(self):

        self.t1.update_threshold(0.0)
        self.t2.update_threshold(0.0)

        metadata1 = read_pickle(
            "C:/Users/bscuser/Projects/FastComposedModels/Definitions/Classifiers/sota_models_cifar10-32-dev_validation/V001_DenseNet121_ref_0.pkl")
        metadata2 = read_pickle(
            "C:/Users/bscuser/Projects/FastComposedModels/Definitions/Classifiers/sota_models_cifar10-32-dev_validation/V001_DenseNet121_ref_0.pkl")
        metadata3 = read_pickle(
            "C:/Users/bscuser/Projects/FastComposedModels/Definitions/Classifiers/sota_models_cifar10-32-dev_validation/V001_MobileNetV2_ref_0.pkl")

        ids = metadata1["test"]["id"]
        inference_t = metadata1["metrics"]["time"]/128.0 + \
                     metadata2["metrics"]["time"]/128.0 + \
                     metadata3["metrics"]["time"]/128.0
        true_time = len(ids) * inference_t
        input = torch.range(0, len(ids) - 1).long()
        self.ensemble(input)
        self.assertEqual(self.ensemble.get_processing_time(), true_time)

    def test_time_threshold_1(self):

        self.t1.update_threshold(1.1)
        self.t2.update_threshold(1.1)

        metadata1 = read_pickle(
            "C:/Users/bscuser/Projects/FastComposedModels/Definitions/Classifiers/sota_models_cifar10-32-dev_validation/V001_DenseNet121_ref_0.pkl")

        ids = metadata1["test"]["id"]
        inference_t = metadata1["metrics"]["time"]/128.0
        true_time = len(ids) * inference_t
        input = torch.range(0, len(ids) - 1).long()
        self.ensemble(input)
        self.assertEqual(self.ensemble.get_processing_time(), true_time)


if __name__ == '__main__':
    unittest.main()
