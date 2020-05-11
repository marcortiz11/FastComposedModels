import source.system_builder_serializable as sb
import source.make_util as make
import source.io_util as io
import os

if __name__ == "__main__":

    dset = "sota_models_cifar10-32-dev_validation"
    Classifier_Path = os.path.join(os.environ['FCM'], 'definitions', 'Classifiers', dset)

    P = []
    models = [f for f in os.listdir(Classifier_Path)]

    # Creating system
    records = {}
    for m_ in models:
        # Model 2
        sys = sb.SystemBuilder(verbose=False)
        classifier = make.make_classifier(m_, os.path.join(Classifier_Path, m_))
        sys.add_classifier(classifier)
        P.append(sys)

    io.save_pickle('initial_population', P)

