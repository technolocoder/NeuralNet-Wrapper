#include "CL.hpp"
#include "activations.hpp"
#include <cstring>
#include <cmath>

TEMPLATE void _CL<T>::init(int _input_width, int _input_height, int _filter_width, int _filter_height ,int _stride_width, int _stride_height ,int _input_depth, int _output_depth, int _sample_size, ACTIVATION_TYPES _activation){
    input_width = _input_width;
    input_height = _input_height;
    filter_width = _filter_width;
    filter_height = _filter_height;
    stride_width = _stride_width;
    stride_height = _stride_height;
    activation = _activation;
    input_depth = _input_depth;
    output_depth = _output_depth;
    sample_size = _sample_size;
    output_width = (input_width-filter_width)/stride_width+1;
    output_height = (input_height-filter_height)/stride_height+1;
    kernel_size = filter_width*filter_height*input_depth*output_depth;
    output_size = output_width*output_height*output_depth*sample_size;
    total = kernel_size+output_size+output_depth;
    preset = false;
}

TEMPLATE void _CL<T>::process(){
    int output_index = 0;
    memset(o_ptr,0,sizeof(T)*output_size);
    for(int i = 0; i < sample_size; ++i){
        for(int j = 0; j < output_depth; ++j){
            for(int k = 0; k < input_depth; ++k){
                int index = 0;
                for(int l = 0; l <= input_height-filter_height; l += stride_height){
                    for(int h = 0; h <= input_width-filter_width; h += stride_width){
                        T sum = 0;
                        for(int y = 0; y < filter_height; ++y){
                            for(int x = 0; x < filter_width; ++x){
                                sum += i_ptr[i*input_depth*input_width*input_height+k*input_width*input_height+(l+y)*input_width+h+x]*k_ptr[j*filter_width*filter_height*input_depth+k*filter_width*filter_height+y*filter_width+x];
                            }
                        }
                        o_ptr[index++ + output_index * output_width*output_height] += sum;
                    }
                }
            }
            switch(activation){
            case TANH:
                for(int l = 0; l < output_width*output_height; ++l) o_ptr[output_index * output_width*output_height + l] = tanh(o_ptr[output_index * output_width*output_height + l]+b_ptr[j]); 
                break;
            case RELU:
                for(int l = 0; l < output_width*output_height; ++l) o_ptr[output_index * output_width*output_height + l] = relu(o_ptr[output_index * output_width*output_height + l]+b_ptr[j]);
                break;
            case LEAKY_RELU:
                for(int l = 0; l < output_width*output_height; ++l) o_ptr[output_index * output_width*output_height + l] = leaky_relu(o_ptr[output_index * output_width*output_height + l]+b_ptr[j]);
                break;
            case SIGMOID:
                for(int l = 0; l < output_width*output_height; ++l) o_ptr[output_index * output_width*output_height + l] = sigmoid(o_ptr[output_index * output_width*output_height + l]+b_ptr[j]);
                break;
            case LINEAR:
                for(int l = 0; l < output_width*output_height; ++l) o_ptr[output_index * output_width*output_height + l] = o_ptr[output_index * output_width*output_height + l]+b_ptr[j];
                break;
            }
            ++output_index;

        }
    }       
}
template struct _CL<float>;
template struct _CL<double>;