import Source.system_builder as sb
import Source.protobuf.make_util as make
import Source.system_evaluator as eval

if __name__ == "__main__":

    Classifier_Path = "../../Definitions/Classifiers/"
    classifier_file = "DenseNet121_cifar10.pkl"

    # Creating system
    sys = sb.SystemBuilder(verbose=False)
    smallClassifier = make.make_classifier("Classifier", Classifier_Path+classifier_file)
    sys.add_classifier(smallClassifier)
    results = eval.evaluate(sys, "Classifier", check_classifiers=True)
    eval.pretty_print(results)

