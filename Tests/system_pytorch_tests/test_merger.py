import unittest
import torch
from Source.pytorch.classifier_metadata import ClassifierMetadata, Split
from Source.pytorch.merger import Merger, MergeProtocol
from Source.pytorch.system import System
from Source.io_util import read_pickle


class TestMerger(unittest.TestCase):

    def setUp(self) -> None:
        self.c1 = ClassifierMetadata(
            "C:/Users/bscuser/Projects/FastComposedModels/Definitions/Classifiers/sota_models_cifar10-32-dev_validation/V001_DenseNet121_ref_0.pkl")
        self.c2 = ClassifierMetadata(
            "C:/Users/bscuser/Projects/FastComposedModels/Definitions/Classifiers/sota_models_cifar10-32-dev_validation/V001_DenseNet121_ref_0.pkl")
        self.c3 = ClassifierMetadata(
            "C:/Users/bscuser/Projects/FastComposedModels/Definitions/Classifiers/sota_models_cifar10-32-dev_validation/V001_MobileNetV2_ref_0.pkl")
        self.merger = Merger([self.c1, self.c2, self.c3], MergeProtocol.VOTING)
        self.ensemble = System(self.merger, Split.TEST)

    def test_merger_voting(self):
        # Inference on sample 0 of test split
        input = torch.tensor([0])
        pred = self.ensemble(input)
        print(pred)
        print(self.c1(input).argmax(dim=1))
        print(self.c2(input).argmax(dim=1))
        print(self.c3(input).argmax(dim=1))
        gt = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
        self.assertEqual(True, torch.equal(pred, gt))

    def test_merger_max(self):
        # Inference on sample 0 of test split
        self.merger.update_merge_protocol(MergeProtocol.MAX)
        input = torch.tensor([0])
        pred = self.ensemble(input)
        print(pred)
        print(self.c1(input))
        print(self.c2(input))
        print(self.c3(input))
        gt = self.c3(input)
        self.assertEqual(True, torch.equal(pred, gt))


if __name__ == '__main__':
    unittest.main()
