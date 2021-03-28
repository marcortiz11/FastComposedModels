import Source.pytorch as systorch
from Data.datasets import Split
from Source.pytorch.genetic_operations.crossover import single_point_crossover
import unittest
import os


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:

        # First ensemble
        self.c1 = systorch.ClassifierMetadata(
            os.environ[
                "FCM"] + "/Definitions/Classifiers/sota_models_cifar10-32-dev_validation/V001_DenseNet121_ref_0.pkl")
        self.c2 = systorch.ClassifierMetadata(
            os.environ[
                "FCM"] + "/Definitions/Classifiers/sota_models_cifar10-32-dev_validation/V001_DenseNet121_ref_0.pkl")
        self.t1 = systorch.Trigger(0.0)
        self.chain = systorch.Chain([self.c1, self.t1, self.c2])
        self.merger = systorch.Merger([self.chain])
        self.system = systorch.System(self.merger, Split.TEST)

        # Second ensemble
        self.c3 = systorch.ClassifierMetadata(
            os.environ[
                "FCM"] + "/Definitions/Classifiers/sota_models_cifar10-32-dev_validation/V001_PreActResNet34_ref_0.pkl")
        self.chain2 = systorch.Chain([self.c3])
        self.system2 = systorch.System(self.chain2, Split.TEST)

    def test_exchange_chains_ensemble(self):
        single_point_crossover(self.system, self.system2, self.chain, self.chain2)
        correct = self.merger.get_merged_modules()[0] == self.chain2
        correct = correct and self.system2.get_start() == self.chain
        self.assertEqual(True, correct)
        

if __name__ == '__main__':
    unittest.main()
