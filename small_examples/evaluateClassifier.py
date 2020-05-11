import source.system_builder as sb
import source.make_util as make
import source.system_evaluator as eval

if __name__ == "__main__":

    Classifier_Path = "../definitions/Classifiers/"
    classifier_file = "DenseNet121_cifar10.pkl"

    # Creating system
    sys = sb.SystemBuilder(verbose=False)
    smallClassifier = make.make_classifier("Classifier", Classifier_Path+classifier_file)
    sys.add_classifier(smallClassifier)
    results = eval.evaluate(sys, "Classifier", check_classifiers=True)
    eval.pretty_print(results)

