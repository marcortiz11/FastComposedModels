import numpy as np
import source.make_util as make_util
import source.io_util as io_util

def get_dummy_ClassifierRawData(num_c=3, n=5):
    logits = np.random.rand(n,num_c)
    gt = np.random.randint(0,num_c,n)
    id = np.arange(n)

    # Construct the message
    message = make_util.make_classifier_raw_data(logits, gt, id)
    return message


if __name__ == '__main__':
    # Construct a fake classifier
    msg_train = get_dummy_ClassifierRawData(n=5)
    msg_test = get_dummy_ClassifierRawData(n=4)
    id = "SimpleClassifier001"
    data_id = "SimpleFakeDataset"
    params=1e5
    flops=1e15
    perf_time_mean=0.01012
    msg_metrics = make_util.make_performance_metrics(perf_time_mean, int(params), int(flops))

    # def make_classifier(id, train, test, data, metrics=None):
    msg = make_util.make_classifier(id, msg_train, msg_test, data_id, msg_metrics)

    print(msg)
    # # Save as demo file
    io_util.save_message("demo_classifier", msg)