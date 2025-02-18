#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include "Layer.h"

class NeuralNetwork {
private:
    std::vector<Layer> layers;  // Stores all layers of the network
    double learning_rate;       // Learning rate for weight updates

public:
    NeuralNetwork(std::vector<int> layer_sizes, double learning_rate);

    std::vector<double> forward(const std::vector<double>& input);
    void backward(const std::vector<double>& expected);
    void train(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y, int epochs);
};

#endif // NEURAL_NETWORK_H
