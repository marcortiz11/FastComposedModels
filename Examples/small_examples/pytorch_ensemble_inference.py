import torch
from Data.datasets import Cifar10, get_data_loader
from Source.iotnets.networks.networks import build
import os

# Load data
dataset = Cifar10(os.path.join(os.environ['FCM'], 'Data', 'cifar10'), train=False)  # Test set
dloader = get_data_loader(dataset)
print("Instantiated data loader")

# Build model
model = build('DenseNet169', num_c=len(dataset.classes))
print("Built model")

# Load checkpoint
checkpoint = torch.load(os.path.join(os.environ['FCM'],
                                              'Definitions',
                                              'Checkpoints',
                                              'sota_models_cifar10-32-dev',
                                              'V001_DenseNet169_ref_0.t7'),
                                 map_location=torch.device('cpu'))

keys_dict = list(checkpoint['net'].keys())
for key in keys_dict:
    checkpoint['net'][key[7:]] = checkpoint['net'][key]
    del checkpoint['net'][key]

model.load_state_dict(checkpoint['net'])
print("Loaded checkpoint")


"""
8, 4, 4, 3, 5, 5, 2, 5, 8, 2, 3, 6, 9, 7, 6, 4, 9, 7, 4, 5, 7, 3, 1, 6,
        9, 0, 1, 3, 6, 5, 5, 9, 1, 9, 6, 0, 1, 2, 6, 9, 1, 5, 9, 5, 3, 6, 8, 4,
        7, 5, 6, 4, 3, 8, 4, 4, 7, 7, 7, 5, 7, 9, 5, 8, 5, 0, 2, 0, 8, 7, 0, 5,
        4, 4, 3, 0, 6, 8, 8, 4, 4, 2, 2, 7, 0, 9, 0, 7, 4, 6, 3, 8, 4, 5, 1, 5,
        5, 3, 3, 7, 1, 9, 5, 8, 2, 2, 8, 5, 4, 7, 1, 3, 5, 0, 2, 8, 3, 1, 5, 7,
        4, 8, 2, 0, 4, 3, 4, 8

8, 4, 4, 3, 5, 5, 2, 5, 8, 2, 3, 6, 9, 7, 6, 4, 9, 7, 4, 5, 7, 3, 1, 6,
        9, 0, 1, 3, 6, 5, 5, 9, 1, 9, 6, 0, 1, 6, 6, 9, 1, 5, 9, 5, 3, 6, 8, 2,
        7, 5, 6, 4, 6, 8, 4, 4, 7, 7, 8, 8, 7, 9, 5, 8, 5, 0, 2, 0, 8, 7, 0, 5,
        4, 4, 3, 0, 6, 8, 8, 4, 4, 2, 2, 7, 0, 9, 0, 7, 4, 6, 3, 8, 4, 5, 1, 5,
        5, 3, 3, 7, 1, 9, 5, 8, 2, 2, 8, 5, 4, 7, 1, 3, 5, 0, 2, 8, 3, 1, 0, 7,
        4, 8, 2, 0, 4, 0, 4, 8
        
8, 4, 2, 3, 5, 5, 2, 5, 8, 2, 3, 6, 9, 7, 6, 4, 9, 7, 4, 5, 7, 3, 5, 6,
        9, 0, 1, 3, 6, 5, 5, 9, 1, 9, 6, 0, 1, 6, 6, 9, 1, 5, 9, 5, 3, 6, 8, 3,
        7, 5, 6, 4, 9, 8, 4, 4, 7, 7, 8, 8, 7, 9, 5, 8, 5, 0, 2, 0, 8, 7, 0, 5,
        4, 4, 3, 0, 6, 8, 8, 4, 4, 2, 2, 7, 0, 9, 0, 7, 4, 6, 3, 8, 4, 5, 1, 5,
        5, 3, 3, 7, 1, 9, 5, 8, 2, 2, 8, 5, 4, 7, 1, 3, 5, 0, 2, 8, 3, 1, 0, 7,
        4, 8, 2, 0, 4, 0, 4, 8 
        
        
import numpy as np
from PIL import Image
image = dataset[0][0]
image[0, :, :] = (image[0, :, :] + 0.4914) * 0.2023
image[1, :, :] = (image[1, :, :] + 0.4822) * 0.1994
image[2, :, :] = (image[2, :, :] + 0.4465) * 0.2010
image *= 255.0
image = ((image.numpy()).transpose((1, 2, 0))).astype(np.uint8)
image = Image.fromarray(image, 'RGB')
gt = dataset[0][1]
print(f"True:{dataset.classes[gt]}, Predicted:{dataset.classes[8]}")
image.show()


"""


# Trying inference
model = model.eval()
with torch.no_grad():
    for (input, target) in dloader:
        print(input.shape)
        predictions = model(input)
        print(predictions.argmax(dim=1))
        print(target)
        print("\n\n")
