"""
    Definition of methods that build a protobuf message from raw Data
"""
import Source.FastComposedModels_pb2 as fcm
import numpy as np


def make_source(train_path, test_path, format, val_path=None):
    assert "/Data" in test_path or "\Data" in test_path and \
           "/Data" in train_path or "\Data" in train_path,\
        "Datasets should be saved in the Data folder of the project."
    message = fcm.Data.Source(train_path=train_path, test_path=test_path, val_path=val_path, format=format)
    return message


def make_data(id, num_train, num_test, label_name=None, source=None):
    message = fcm.Data()
    message.id = id
    message.num_train = num_train
    message.num_test = num_test
    if source is not None:
        message.source.CopyFrom(source)
    if label_name is not None:
        message.label_name.extend(label_name)
    return message


def make_performance_metrics(**args):
    performance = {
        'time': args['time'] if 'time' in args else 0,
        'ops': args['ops'] if 'ops' in args else 0,
        'params': args['params'] if 'params' in args else 0,
    }
    performance['times'] = args['times'] if 'times' in args else [performance['time']]
    return performance


def make_classifier_raw_data(L, gt, ids):
    raw_data = {
        'logits': np.array(L),
        'gt': np.array(gt),
        'id': np.array(ids),
    }
    return raw_data


def make_classifier_dict(name, data_id, train_data, test_data, performance, val_data=None, params=None):

    classifier = {
        'name': name,
        'data_id': data_id,
        'train': train_data,
        'test': test_data,
        'val': val_data,
        'metrics': performance,
        'optional_params': params,
    }

    return classifier


def make_classifier(id, classifier_file, component_id=None, data_id=None):
    message = fcm.Classifier()
    message.id = id
    message.classifier_file = classifier_file
    if data_id is not None:
        message.data_id = data_id
    if component_id is not None:
        message.component_id = component_id
    return message


def make_empty_classifier(id="", data_id=None, component_id=None):
    message = fcm.Classifier()
    message.id = id  # id is mandatory
    if component_id is not None:
        message.component_id = component_id
    if data_id is not None:
        message.data_id = data_id
    return message


def make_trigger(id, classifier, component_ids, model=None):
    message = fcm.Trigger()
    message.id = id
    if model is not None:
        message.model = model
    message.classifier.CopyFrom(classifier)  # Classifier has to be trained
    for component_id in component_ids:
        message.component_ids.append(component_id)
    return message


def make_empty_trigger(id, component_ids):
    message = fcm.Trigger()
    message.id = id
    for component_id in component_ids:
        message.component_ids.append(component_id)
    return message


def make_merger(id, merged_ids, merge_type=None):
    message = fcm.Merger()
    message.id = id
    for c in merged_ids:
        message.merged_ids.append(c)
    if merge_type is not None:
        message.merge_type = merge_type
    return message
