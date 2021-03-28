#ifndef __SL_WRAPPER__
#define __SL_WRAPPER__
#include "common.hpp"

TEMPLATE struct _SL{
    void init(int _layer_size,int _sample_size);
    void process();

    int sample_size, layer_size,total;
    T *i_ptr, *o_ptr;
};

#endif 