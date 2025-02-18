#ifndef LAYER_H
#define LAYER_H

#include <vector>

class Layer {
public:
    int num_inputs;               // Number of inputs to the layer
    int num_neurons;              // Number of neurons in the layer
    std::vector<std::vector<double>> weights;  // Weights matrix
    std::vector<double> biases;    // Biases for each neuron
    std::vector<double> outputs;   // Activated outputs of the layer
    std::vector<double> inputs;    // Inputs to the layer

    Layer(int num_inputs, int num_neurons);
    std::vector<double> activate(const std::vector<double>& input);
    void updateWeights(const std::vector<double>& d_weights, const std::vector<double>& d_biases, double learning_rate);
};

#endif // LAYER_H
