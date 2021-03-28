#include "SL.hpp"
#include <cmath>

TEMPLATE void _SL<T>::init(int _layer_size ,int _sample_size){
    layer_size = _layer_size;
    sample_size = _sample_size;
    total = layer_size*sample_size;
}

TEMPLATE void _SL<T>::process(){
    for(int i = 0; i < sample_size; ++i){
        T _max = i_ptr[i*layer_size],sum = 0;
        for(int j = 1; j < layer_size; ++j) _max = MAX(i_ptr[i*layer_size+j],_max);
        for(int j = 0; j < layer_size; ++j) sum += exp(i_ptr[i*layer_size+j]-_max);
        for(int j = 0; j < layer_size; ++j) o_ptr[i*layer_size+j] = exp(i_ptr[i*layer_size+j]-_max)/sum;
    }
}

template struct _SL<float>;
template struct _SL<double>;