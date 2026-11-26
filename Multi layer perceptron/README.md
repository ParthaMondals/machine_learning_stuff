# Perceptron and Multilayer Perceptron

This project contains two neural-network implementations:

- A single perceptron trained with the perceptron learning algorithm.
- A multilayer perceptron (MLP) trained with backpropagation.

Both training programs use the sample data defined in their Python files. NumPy is required.

## Single perceptron

The `perceptron` class in [perceptron.py](perceptron.py) initializes one weight per input and a bias. Its `bipolar_forward` method returns `1` when the weighted sum plus bias is non-negative, and `-1` otherwise.

The training program, [train_perceptron.py](train_perceptron.py), trains on four bipolar examples. 

At each epoch, the program prints each sample's prediction and error, then adjusts the weights and bias using the learning rate. It stops when all training samples are classified correctly. After training, it prompts for one test input and its expected output, then prints the prediction.

Run from the project directory:

```bash
python train_perceptron.py
```

Enter three numeric input values separated by spaces when prompted for a test sample.

### Perceptron experiments

#### Experiment 1

- Learning rate: 1
- Input data: 
[
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
]
- Target data: [1, 0,0,0,0,0]
- Training output: 
Enter the learning rate: 1
Initial weight: [0.375 0.951 0.192] initial bias: 0.241
Epoch: 0 input: [0 0 0] output: 1 Error: 0
Epoch: 0 input: [0 0 1] output: 1 Error: -2
Epoch: 0 input: [0 1 0] output: -1 Error: 0
Epoch: 0 input: [0 1 1] output: -1 Error: 0
Epoch: 0 input: [1 0 0] output: -1 Error: 0
Epoch: 0 input: [1 0 1] output: -1 Error: 0
Epoch: 1 input: [0 0 0] output: -1 Error: 2
Epoch: 1 input: [0 0 1] output: -1 Error: 0
Epoch: 1 input: [0 1 0] output: 1 Error: -2
Epoch: 1 input: [0 1 1] output: -1 Error: 0
Epoch: 1 input: [1 0 0] output: -1 Error: 0
Epoch: 1 input: [1 0 1] output: -1 Error: 0
Epoch: 2 input: [0 0 0] output: -1 Error: 2
Epoch: 2 input: [0 0 1] output: -1 Error: 0
Epoch: 2 input: [0 1 0] output: -1 Error: 0
Epoch: 2 input: [0 1 1] output: -1 Error: 0
Epoch: 2 input: [1 0 0] output: 1 Error: -2
Epoch: 2 input: [1 0 1] output: -1 Error: 0
Epoch: 3 input: [0 0 0] output: -1 Error: 2
Epoch: 3 input: [0 0 1] output: -1 Error: 0
Epoch: 3 input: [0 1 0] output: -1 Error: 0
Epoch: 3 input: [0 1 1] output: -1 Error: 0
Epoch: 3 input: [1 0 0] output: -1 Error: 0
Epoch: 3 input: [1 0 1] output: -1 Error: 0
Epoch: 4 input: [0 0 0] output: 1 Error: 0
Epoch: 4 input: [0 0 1] output: -1 Error: 0
Epoch: 4 input: [0 1 0] output: -1 Error: 0
Epoch: 4 input: [0 1 1] output: -1 Error: 0
Epoch: 4 input: [1 0 0] output: -1 Error: 0
Epoch: 4 input: [1 0 1] output: -1 Error: 0
Training Complete! at iteration: 5

- Testing input and expected target:
Input= [1, 1. 1]
targets = 0
- Testing output:

Final weight: [-1.625 -1.049 -1.808] Final bias: 0.2410000000000001
Lets begin Testing- 

Enter all bits (separated by space): 1 1 1
Enter expected output: 0
Sample is: ['1' '1' '1']
Target: 0 Predicted: 0
Enter all bits (separated by space): 1 0 0
Enter expected output: 0 
Sample is: ['1' '0' '0']
Target: 0 Predicted: 0
Enter all bits (separated by space): 


## Multilayer perceptron

The `MLP` and `Layer` classes in [network.py](network.py) build a feed-forward network from perceptrons with sigmoid activations. The network's layer sizes are selected interactively when training starts. Backpropagation computes gradients and updates weights and biases using gradient descent, one training sample at a time.

[train_mlp.py](train_mlp.py) provides all eight combinations of three binary inputs and their odd-parity targets:

| Input | Target |
| --- | ---: |
| `[0, 0, 0]` | `0` |
| `[0, 0, 1]` | `1` |
| `[0, 1, 0]` | `1` |
| `[0, 1, 1]` | `0` |
| `[1, 0, 0]` | `1` |
| `[1, 0, 1]` | `0` |
| `[1, 1, 0]` | `0` |
| `[1, 1, 1]` | `1` |

The training program asks for the learning rate, precision, number and size of hidden layers, and output-layer size. It prints training predictions every 100 iterations and when training completes. Completion is determined by the output gradient for the last sample in the most recent pass falling below the entered precision.

After training, the program saves the serialized model as `model.pkl` in the current directory. [test_mlp.py](test_mlp.py) loads that file and runs a prediction for `[1, 0, 0]`.

Run the training program first, then test the saved model from the same directory:

```bash
python train_mlp.py
python test_mlp.py
```

### MLP experiments

#### Experiment 1

- Learning rate: 0.1
- Precision: 0.02
- Hidden-layer count and sizes: 1 hidden layer with 4 perceptron
- Output-layer size: 1
- Input data:
     [0.1, 0.2],  # 0
    [0.2, 0.8],  # 1
    [0.8, 0.2],  # 1
    [0.9, 0.8],  # 0
    [0.3, 0.4],  # 0
    [0.4, 0.7],  # 1
    [0.7, 0.3],  # 1
    [0.6, 0.9],  # 0
The rule is: target 1 when one input is below 0.5 and the other is above 0.5; otherwise target 0.
- Target data:
 [ [0], [1], [1], [0],
    [0], [1], [1], [0], ]

- Training output:

Iteration: 161600
Input [0.1 0.2] -> output [0.00230198], target [0.], error [0.02000935]
Input [0.2 0.8] -> output [0.9986998], target [1.], error [0.02000935]
Input [0.8 0.2] -> output [0.99837211], target [1.], error [0.02000935]
Input [0.9 0.8] -> output [0.00145347], target [0.], error [0.02000935]
Input [0.3 0.4] -> output [0.00844948], target [0.], error [0.02000935]
Input [0.4 0.7] -> output [0.98808416], target [1.], error [0.02000935]
Input [0.7 0.3] -> output [0.99673858], target [1.], error [0.02000935]
Input [0.6 0.9] -> output [0.0100015], target [0.], error [0.02000935]

Iteration: 161700
Input [0.1 0.2] -> output [0.00230103], target [0.], error [0.02000192]
Input [0.2 0.8] -> output [0.9987001], target [1.], error [0.02000192]
Input [0.8 0.2] -> output [0.99837271], target [1.], error [0.02000192]
Input [0.9 0.8] -> output [0.0014529], target [0.], error [0.02000192]
Input [0.3 0.4] -> output [0.00844652], target [0.], error [0.02000192]
Input [0.4 0.7] -> output [0.98808855], target [1.], error [0.02000192]
Input [0.7 0.3] -> output [0.99673987], target [1.], error [0.02000192]
Input [0.6 0.9] -> output [0.00999779], target [0.], error [0.02000192]

Training complete.
**Training took 161700 Iteration to finish**
- Testing input:
    [0.15, 0.75],  # 1
    [0.75, 0.15],  # 1
    [0.85, 0.65],  # 0
    [0.25, 0.35],  # 0 
- Expected target:
    [[1], [1], [0], [0]]

- Testing output:

Input [0.15 0.75] -> output 1 : Expected [1.]
Input [0.75 0.15] -> output 1 : Expected [1.]
Input [0.85 0.65] -> output 0 : Expected [0.]
Input [0.25 0.35] -> output 0 : Expected [0.]

