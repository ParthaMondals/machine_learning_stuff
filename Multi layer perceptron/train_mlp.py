import numpy as np
from network import MLP
import pickle

## Define the training data for the model
inputs = np.array([
     [0.1, 0.2],  # 0
    [0.2, 0.8],  # 1
    [0.8, 0.2],  # 1
    [0.9, 0.8],  # 0
    [0.3, 0.4],  # 0
    [0.4, 0.7],  # 1
    [0.7, 0.3],  # 1
    [0.6, 0.9],  # 0
], dtype=float)

## Define the target outputs for the training data
targets = np.array([
     [0], [1], [1], [0],
    [0], [1], [1], [0],
], dtype=float)



model = MLP()  ## Create an instance of the MLP class to represent the neural network

learning_rate, precision = model.initiate_mlp(inputs)  ## Initialize the model and get the learning rate and precision for training

model.train_mlp(inputs, targets, learning_rate, precision) ## Train the model with the provided inputs, targets, learning rate, and precision

## Save the trained model to a file using pickle
with open("model.pkl", "wb") as f:     # wb = write binary
    pickle.dump(model, f)
    
