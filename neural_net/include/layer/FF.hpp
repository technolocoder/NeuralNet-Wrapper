#ifndef __FF_WRAPPER__
#define __FF_WRAPPER__

#include "common.hpp"

TEMPLATE struct _FF{
    void init(int _input_size ,int _output_size, int _sample_size, ACTIVATION_TYPES _activation);
    void process();

    int input_size,output_size,sample_size;
    ACTIVATION_TYPES activation;
    T *i_ptr, *o_ptr, *weights, *bias;
};

#endif 