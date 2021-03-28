#include "FC.hpp"
#include "activations.hpp"

TEMPLATE void _FC<T>::init(int _input_size ,int _layer_size ,int _sample_size ,ACTIVATION_TYPES _activation){
    input_size = _input_size;
    layer_size = _layer_size;
    sample_size = _sample_size;
    activation = _activation;
    weight_count = input_size*layer_size;
    total_output_size = layer_size*sample_size;
    total = weight_count+total_output_size;
}

#define LOOP for(int i = 0; i < total_output_size; ++i)

TEMPLATE void _FC<T>::process(){
    int index = 0;
    for(int i = 0; i < sample_size; ++i){
        for(int j = 0; j < layer_size; ++j){
            T sum = 0;
            for(int k = 0; k < input_size; ++k) sum += w_ptr[j*input_size+k]*i_ptr[i*input_size+k];
            o_ptr[index++] = sum+b_ptr[j];
        }
    }
    
    if(activation == TANH) LOOP o_ptr[i] = tanh(o_ptr[i]);
    else if(activation == SIGMOID) LOOP o_ptr[i] = sigmoid(o_ptr[i]);
    else if(activation == RELU) LOOP o_ptr[i] = relu(o_ptr[i]);
    else if(activation == LEAKY_RELU) LOOP o_ptr[i] = leaky_relu(o_ptr[i]);
}

template struct _FC<float>;
template struct _FC<double>;