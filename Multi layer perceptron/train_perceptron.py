import numpy as np
import pickle
from perceptron import perceptron

inputs_nand = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
])

targets = np.array([1, 0,0,0,0,0])
bipolar_targets = np.where(targets == 0, -1, 1)

p = perceptron(inputs_nand.shape[1])  # 3 inputs


## Training
learning_rate = float(input("Enter the learning rate: "))
print(f"Initial weight: {p.weights} initial bias: {p.bias}")
p.perceptron_training(inputs_nand,bipolar_targets ,learning_rate)
print(f"Final weight: {p.weights} Final bias: {p.bias}")

## Testing
print("Lets begin Testing- \n")
while(True):
    bits = input("Enter all bits (separated by space): ")
    if(bits == 'exit'):
        break
    value = int(input("Enter expected output: "))

    sample = np.array(bits.split())
    print(f"Sample is: {sample}")

    prediction = p.bipolar_forward(sample)
    print(f"Target: {value} Predicted: {0 if prediction == -1 else 1}")