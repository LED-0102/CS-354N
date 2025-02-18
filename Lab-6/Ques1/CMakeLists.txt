cmake_minimum_required(VERSION 3.10)
project(Ques1)

set(CMAKE_CXX_STANDARD 17)

# Include the 'include' directory for header files
include_directories(include)

# Add executable and source files
add_executable(Ques1
        src/main.cpp
        src/NeuralNetwork.cpp
        src/Layer.cpp
        src/Activation.cpp
)
