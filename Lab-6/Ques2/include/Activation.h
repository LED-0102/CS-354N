#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <vector>
#include <cmath>

namespace Activation {
    double sigmoid(double x);
    double sigmoid_derivative(double x);
    std::vector<double> apply_sigmoid(const std::vector<double>& values);
    std::vector<double> apply_sigmoid_derivative(const std::vector<double>& values);
}

#endif // ACTIVATION_H
