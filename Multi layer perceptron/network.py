import numpy as np
from perceptron import perceptron

class Layer:
    def __init__(self, perceptron_count, input_size):
        self.perceptrons = [perceptron(input_size) for _ in range(perceptron_count)]
        ## perceptrons is a list of perceptron objects, each representing a single perceptron in the layer
    
    def forward_pass(self, inputs):
        outputs = []
        for p in self.perceptrons:
            output = p.forward(inputs)
            outputs.append(output)
        return np.array(outputs)  # Return the outputs as a numpy array
        # Size of this output array will be equal to the number of perceptrons in the layer
    
    def backward_pass(self, grad_outputs):
        # Accumulate gradients passed back to this layer's inputs.
        grad_inputs = np.zeros_like(self.perceptrons[0].last_input)
        weight_gradients = []
        bias_gradients = []

        # Each output gradient corresponds to one perceptron in this layer.
        for p, grad_output in zip(self.perceptrons, grad_outputs):
            input_grad, weight_grad, bias_grad = p.backward(grad_output)
            grad_inputs += input_grad
            weight_gradients.append(weight_grad)
            bias_gradients.append(bias_grad)

        # Keep parameter gradients for the weight-update step.
        self.grad_weights = np.array(weight_gradients)
        self.grad_biases = np.array(bias_gradients)
        return grad_inputs


class MLP:
    def __init__(self):
        self.layers = []

    def add_layer(self, perceptron_count, input_size):
        layer = Layer(perceptron_count, input_size)
        self.layers.append(layer)

    def predict(self, inputs):
        inputs = np.asarray(inputs, dtype=float)        
        for layer in self.layers:
            outputs = layer.forward_pass(inputs)
            inputs = outputs  # The output of the current layer becomes the input for the next layer

        return outputs  # Return the final output after passing through all layers

    def backward_propagate(self, grad_output):
        # Propagate gradients backward through the layers in reverse order.
        for layer in reversed(self.layers):
            grad_output = layer.backward_pass(grad_output)
        return grad_output

    def initiate_mlp(self,inputs):
        ## Prompt the user to enter the learning rate for training the model
        learning_rate = float(input("Enter the learning rate: "))
        input_layer_size = inputs.shape[1]
        precision = float(input("Enter the precision for training completion: "))  ## Set the precision for training completion

        ## Prompt the user to enter the number of hidden layers and their sizes
        no_of_hidden_layers = int(input("Enter the number of hidden layers: "))
        hidden_sizes = []
        for i in range(no_of_hidden_layers): ### Create the hidden layers based on user input
            hidden_sizes.append(
                int(input(f"Enter the number of perceptrons in hidden layer {i + 1}: "))
            )

        ## Add the hidden layers to the model with the specified sizes
        for hidden_size in hidden_sizes:
            self.add_layer(hidden_size, input_layer_size)
            input_layer_size = hidden_size

        output_layer_size = int(input("Enter the number of perceptrons in the output layer: ")) 
        self.add_layer(output_layer_size, input_layer_size) ## Add the output layer to the model with the specified size

        return learning_rate, precision  ## Return the learning rate and precision for training the model

    def update_parameters(self, learning_rate):
        # Adjust each perceptron's weights and bias using its calculated gradients.
        for layer in self.layers:
            for p, grad_weights, grad_bias in zip(
                layer.perceptrons, layer.grad_weights, layer.grad_biases
            ):
                p.weights -= learning_rate * grad_weights
                p.bias -= learning_rate * grad_bias

    def train_mlp(self, inputs, targets, learning_rate, precision):
        count = 0
        while True:
            for x, y in zip(inputs, targets):
                final_output = self.predict(x)
                grad_output = 2 * (final_output - y)

                self.backward_propagate(grad_output)
                self.update_parameters(learning_rate)

            if (count + 1) % 100 == 0:
                print(f"\nIteration: {count + 1}")
                for x, y in zip(inputs, targets):
                    print(f"Input {x} -> output {self.predict(x)}, target {y}, error {np.abs(grad_output)}")
            count += 1


            if (np.abs(grad_output) < precision).all():
                print("\nTraining complete.")
                break

