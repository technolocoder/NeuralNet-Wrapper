#ifndef __ACTIVATIONS_WRAPPER__
#define __ACTIVATIONS_WRAPPER__

#include "common.hpp"
#include <cmath>

TEMPLATE inline T sigmoid(T sum){
    return 1.0/(1.0+exp(-sum));
}

TEMPLATE inline T relu(T sum){
    return sum>0?sum:0;
}

TEMPLATE inline T leaky_relu(T sum){
    return sum>0?sum:sum*0.01;
}

TEMPLATE inline T sigmoid_deriv(T sigmoid_out){
    return sigmoid_out*(1.0-sigmoid_out);
}

TEMPLATE inline T tanh_deriv(T tanh_out){
    return 1.0-tanh_out*tanh_out;
}

TEMPLATE inline T relu_deriv(T sum){
    return sum>0?1:0;
}

TEMPLATE inline T leaky_relu_deriv(T sum){
    return sum>0?1:0.01;
}

#endif