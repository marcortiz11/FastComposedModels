import Source.make_util as make
import Source.system_evaluator as seval
import Source.system_builder_serializable as sb
import Source.FastComposedModels_pb2 as fcm
from Source.system_evaluator import evaluate
from Source.system_evaluator_utils import pretty_print
import os

classifiers = [os.path.join(os.environ['FCM'], 'Definitions/Classifiers/sota_models_cifar10-32-dev_validation', 'V001_ResNet18_ref_0.pkl'),
               os.path.join(os.environ['FCM'], 'Definitions/Classifiers/sota_models_cifar10-32-dev_validation', 'V001_ResNet152_ref_0.pkl')]


# Build the system
merged_classifiers = sb.SystemBuilder()
merger = make.make_merger('Merger', classifiers, fcm.Merger.AVERAGE)
merged_classifiers.add_merger(merger)
for classifier in classifiers:
    c = make.make_classifier(classifier, classifier)
    merged_classifiers.add_classifier(c)

merged_classifiers.set_start('Merger')
R = evaluate(merged_classifiers, merged_classifiers.get_start())
pretty_print(R)


# Manual check
import Source.io_util as io
import numpy as np

c_dict_0 = io.read_pickle(classifiers[0])
c_dict_1 = io.read_pickle(classifiers[1])

gt = c_dict_0['test']['gt']
logits_0 = c_dict_0['test']['logits']
logits_1 = c_dict_1['test']['logits']

average = (logits_0 + logits_1)/2
test_acc = np.sum(np.argmax(average, 1) == gt)/len(gt)

assert test_acc == R.test['system'].accuracy, "ERROR: Merge operation not working"