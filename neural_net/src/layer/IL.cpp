#include "IL.hpp"

TEMPLATE void _IL<T>::init(int _width, int _height, int _depth, int _sample_size, bool _batch_normalized){
    width = _width;
    height = _height;
    depth = _depth;
    batch_normalized = _batch_normalized;    
    sample_size = _sample_size;

    total = batch_normalized*width*height*depth*sample_size;
}

TEMPLATE void _IL<T>::process(){
    if(batch_normalized){
        //TODO Implement Batch Normalization
        return;
    }
    o_ptr = i_ptr;
}

template struct _IL<float>;
template struct _IL<double>;