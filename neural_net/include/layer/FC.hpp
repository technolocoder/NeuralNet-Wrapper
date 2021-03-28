#ifndef __FC_WRAPPER__
#define __FC_WRAPPER__

#include "common.hpp"

TEMPLATE struct _FC{
    void init(int _input_size ,int _layer_size, int _sample_size, ACTIVATION_TYPES _activation);
    void process();

    int input_size,layer_size,sample_size,total,total_output_size,weight_count;
    ACTIVATION_TYPES activation;
    T *i_ptr, *o_ptr, *w_ptr, *b_ptr;
};

#endif 