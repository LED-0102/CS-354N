#include "NeuralNetwork.h"
#include "Activation.h"
#include <iostream>
#include <cassert>

NeuralNetwork::NeuralNetwork(std::vector<int> layer_sizes, const double learning_rate)
    : learning_rate(learning_rate) {
    // Initialize layers
    assert(layer_sizes.size() > 1);  // The network must have at least one hidden layer
    for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
        layers.emplace_back(layer_sizes[i], layer_sizes[i + 1]);
    }
}

// Forward pass: propagate the input through all layers
std::vector<double> NeuralNetwork::forward(const std::vector<double>& input) {
    std::vector<double> output = input;

    for (auto & layer : layers) {
        output = layer.activate(output);  // Propagate through the layer
    }

    return output;
}

// Backpropagation: compute gradients and update weights
void NeuralNetwork::backward(const std::vector<double>& expected) {
    Layer& hidden_layer = layers[0];  // First (hidden) layer
    Layer& output_layer = layers[1];  // Second (output) layer

    std::vector<double> output_errors(output_layer.num_neurons);
    std::vector<double> hidden_errors(hidden_layer.num_neurons);

    // Compute error at the output layer
    for (int j = 0; j < output_layer.num_neurons; ++j) {
        double error = expected[j] - output_layer.outputs[j];
        output_errors[j] = error * Activation::sigmoid_derivative(output_layer.outputs[j]); // Gradient w.r.t output
    }

    // Compute error at the hidden layer
    for (int j = 0; j < hidden_layer.num_neurons; ++j) {
        double error = 0.0;
        for (int k = 0; k < output_layer.num_neurons; ++k) {
            error += output_layer.weights[k][j] * output_errors[k]; // Weighted sum of output errors
        }
        hidden_errors[j] = error * Activation::sigmoid_derivative(hidden_layer.outputs[j]); // Apply activation derivative
    }

    // Update weights and biases of the output layer
    std::vector<double> d_weights_output(output_layer.num_neurons * output_layer.num_inputs);
    std::vector<double> d_biases_output(output_layer.num_neurons);

    for (int j = 0; j < output_layer.num_neurons; ++j) {
        d_biases_output[j] = output_errors[j]; // Bias update
        for (int k = 0; k < output_layer.num_inputs; ++k) {
            d_weights_output[j * output_layer.num_inputs + k] = output_errors[j] * hidden_layer.outputs[k];
        }
    }

    output_layer.updateWeights(d_weights_output, d_biases_output, learning_rate);

    // Update weights and biases of the hidden layer
    std::vector<double> d_weights_hidden(hidden_layer.num_neurons * hidden_layer.num_inputs);
    std::vector<double> d_biases_hidden(hidden_layer.num_neurons);

    for (int j = 0; j < hidden_layer.num_neurons; ++j) {
        d_biases_hidden[j] = hidden_errors[j]; // Bias update
        for (int k = 0; k < hidden_layer.num_inputs; ++k) {
            d_weights_hidden[j * hidden_layer.num_inputs + k] = hidden_errors[j] * hidden_layer.inputs[k];
        }
    }

    hidden_layer.updateWeights(d_weights_hidden, d_biases_hidden, learning_rate);
}


// Train the neural network
void NeuralNetwork::train(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < X.size(); ++i) {
            // Perform forward pass
            std::vector<double> output = forward(X[i]);

            // Perform backward pass (backpropagation)
            backward(Y[i]);
        }
    }
}
