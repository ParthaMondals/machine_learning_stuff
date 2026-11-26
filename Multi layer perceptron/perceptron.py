import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def bipolar(x):
    return 1 if x>=0 else -1

class perceptron:
    def __init__(self,input_size):
        self.bias = np.round(np.random.rand(), 3)  # Initialize bias with a random value rounded to 3 decimal places
        self.weights = np.round(np.random.rand(input_size), 3)  # Assuming input_size input nodes for the perceptron

    def forward(self, inputs):
        self.last_input = np.asarray(inputs, dtype=float)  # Store the last input for use in backpropagation
        z = np.dot(self.weights, self.last_input) + self.bias
        self.last_output = sigmoid(z)
        return self.last_output

    def backward(self, grad_output):
        # Apply the sigmoid derivative to get the loss gradient for the weighted sum.
        grad_z = grad_output * self.last_output * (1 - self.last_output)
        # Each weight's gradient is grad_z multiplied by its corresponding input.
        grad_weights = grad_z * self.last_input
        # The bias contributes directly to the weighted sum, so its gradient is grad_z.
        grad_bias = grad_z
        # Propagate the loss gradient back to this perceptron's input values.
        grad_inputs = grad_z * self.weights
        # Return input gradients for the previous layer and parameter gradients for updates.
        return grad_inputs, grad_weights, grad_bias

    def bipolar_forward(self, inputs):
        self.last_input = np.asarray(inputs, dtype=float)  # Store the last input for use in backpropagation
        z = np.dot(self.weights, self.last_input) + self.bias
        self.last_output = bipolar(z)
        return self.last_output

    def perceptron_training(self,inputs, targets, learning_rate):
        epoch = 0
        while(True):
            c=0
            for i in range(len(inputs)):
                error = targets[i] - self.bipolar_forward(inputs[i])
                if error == 0: 
                    c = c + 1
                print(f"Epoch: {epoch} input: {inputs[i]} output: {self.bipolar_forward(inputs[i])} Error: {error}")
                weight_change = learning_rate * error * inputs[i]
                self.weights += weight_change
                self.bias += learning_rate * error  # Update bias as well
            epoch = epoch + 1
            if(c == len(inputs)):
                print(f"Training Complete! at iteration: {epoch}")
                break




