#ifndef __CL_WRAPPER__
#define __CL_WRAPPER__

#include "common.hpp"

TEMPLATE struct _CL{
    int input_width, input_height, filter_width, filter_height, stride_width, stride_height ,output_width, output_height ,input_depth, output_depth, sample_size;
    T *i_ptr ,*o_ptr ,*k_ptr, *temp_ptr, *b_ptr;

    bool preset = false;

    ACTIVATION_TYPES activation;
    int output_size, kernel_size ,total;

    void init(int _input_width, int _input_height, int _filter_width, int _filter_height ,int _stride_width, int _stride_height ,int _input_depth, int _output_depth, int _sample_size, ACTIVATION_TYPES _activation);
    void process();
};

#endif 