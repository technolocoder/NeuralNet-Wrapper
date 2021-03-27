#include "PL.hpp"

TEMPLATE void _PL<T>::init(int _input_width ,int _input_height ,int _filter_width, int _filter_height ,int _stride_width, int _stride_height ,int _depth, int _sample_size, POOLING_TYPE _pooling_type){
    input_width = _input_width;
    input_height = _input_height;
    filter_width = _filter_width;
    filter_height = _filter_height;
    stride_width = _stride_width;
    stride_height = _stride_height;
    output_width = (input_width-filter_width)/stride_width+1;
    output_height = (input_height-filter_height)/stride_height+1;
    depth = _depth;
    sample_size = _sample_size;
    pooling_type = _pooling_type;
    total = depth*sample_size*output_width*output_height;
    filter_size = filter_width*filter_height;
}

TEMPLATE void _PL<T>::process(){
    int offset = 0;
    if(pooling_type == MAX_POOL){
        for(int i = 0; i < sample_size; ++i){
            for(int j = 0; j < depth; ++j){
                for(int k = 0; k <= input_height-filter_height; k += stride_height){
                    for(int h = 0; h <= input_width-filter_width; h += stride_width){
                        T max = -2e7;
                        for(int y = 0; y < filter_height; ++y){
                            for(int x = 0; x < filter_width; ++x){
                                max = MAX(max,i_ptr[i*depth*input_width*input_height+j*input_width*input_height+(k+y)*input_width+h+x]);
                            }
                        }
                        o_ptr[offset++] = max;
                    }
                }
            }
        }
    }else{
        for(int i = 0; i < sample_size; ++i){
            for(int j = 0; j < depth; ++j){
                for(int k = 0; k <= input_height-filter_height; k += stride_height){
                    for(int h = 0; h <= input_width-filter_width; h += stride_width){
                        T sum = 0;
                        for(int y = 0; y < filter_height; ++y){
                            for(int x = 0; x < filter_width; ++x){
                                sum += i_ptr[i*depth*input_width*input_height+j*input_width*input_height+(k+y)*input_width+h+x];
                            }
                        }
                        o_ptr[offset++] = sum/filter_size;
                    }
                }
            }
        }
    }
}

template struct _PL<float>;
template struct _PL<double>;