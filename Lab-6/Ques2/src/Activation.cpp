#include "Activation.h"

namespace Activation {

    // Sigmoid activation function: σ(x) = 1 / (1 + e^(-x))
    double sigmoid(const double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    // Derivative of sigmoid: σ'(x) = σ(x) * (1 - σ(x))
    double sigmoid_derivative(const double x) {
        const double sig = sigmoid(x);
        return sig * (1 - sig);
    }

    // Apply sigmoid to a vector (for layers)
    std::vector<double> apply_sigmoid(const std::vector<double>& values) {
        std::vector<double> result(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            result[i] = sigmoid(values[i]);
        }
        return result;
    }

    // Apply sigmoid derivative to a vector
    std::vector<double> apply_sigmoid_derivative(const std::vector<double>& values) {
        std::vector<double> result(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            result[i] = sigmoid_derivative(values[i]);
        }
        return result;
    }

} // namespace Activation
