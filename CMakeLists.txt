cmake_minimum_required(VERSION 3.1...3.7)
project(NeuralNet-Wrapper VERSION 1.0 DESCRIPTION "basic neural network implementation" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_FLAGS "-Ofast")

add_subdirectory(neural_net)
add_subdirectory(demos)
add_subdirectory(test)