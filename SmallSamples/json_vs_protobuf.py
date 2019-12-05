from google.protobuf.json_format import MessageToJson
import Source.make_util as make
import Source.io_util as io
import time

protoClassifier = make.make_empty_classifier()
start = time.time()
io.read_message("../Definitions/Classifiers/V001_SIMPLENET_CIFAR100", protoClassifier)
print("Protobuf reading time:", time.time()-start)

import json

start = time.time()
with open('JSON.txt') as f:
  data = json.load(f)
print("Json reading time:", time.time()-start)

"""

jsonClassifier = MessageToJson(protoClassifier)
with open('./JSON.txt', 'w') as json_file:
  json.dump(jsonClassifier, json_file)


"""
