#include "Layer.h"
#include "Activation.h"
#include <random>  // For random number generation

Layer::Layer(int num_inputs, int num_neurons)
    : num_inputs(num_inputs), num_neurons(num_neurons) {

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());  // Random number generator (Mersenne Twister)
    std::uniform_real_distribution<> dis(-1.0, 1.0);  // Uniform distribution between -1 and 1

    // Initialize weights randomly between -1 and 1
    weights.resize(num_neurons, std::vector<double>(num_inputs));
    for (int i = 0; i < num_neurons; ++i) {
        for (int j = 0; j < num_inputs; ++j) {
            weights[i][j] = dis(gen); // Generate random weight
        }
    }

    // Initialize biases randomly between -1 and 1
    biases.resize(num_neurons);
    for (double &bias : biases) {
        bias = dis(gen); // Generate random bias
    }
}

// Forward pass: computes weighted sum + applies activation
std::vector<double> Layer::activate(const std::vector<double>& input) {
    inputs = input; // Store input for backpropagation
    outputs.resize(num_neurons);

    for (int i = 0; i < num_neurons; ++i) {
        double sum = biases[i]; // Start with bias
        for (int j = 0; j < num_inputs; ++j) {
            sum += weights[i][j] * input[j];
        }
        outputs[i] = Activation::sigmoid(sum); // Apply sigmoid
    }

    return outputs; // Return activated output
}

// Update weights and biases using gradients
void Layer::updateWeights(const std::vector<double>& d_weights, const std::vector<double>& d_biases, double learning_rate) {
    for (int i = 0; i < num_neurons; ++i) {
        biases[i] -= learning_rate * d_biases[i];
        for (int j = 0; j < num_inputs; ++j) {
            weights[i][j] -= learning_rate * d_weights[i * num_inputs + j];
        }
    }
}
