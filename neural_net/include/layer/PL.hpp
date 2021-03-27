#ifndef __PL_WRAPPER__
#define __PL_WRAPPER__

#include "common.hpp"

TEMPLATE struct _PL{
    void init(int _input_width ,int _input_height ,int _filter_width, int _filter_height ,int _stride_width, int _stride_height ,int _depth, 
    int _sample_size, POOLING_TYPE _pooling_type);

    void process();

    int input_width ,input_height ,filter_width ,filter_height ,stride_width, stride_height ,output_width ,output_height ,depth, sample_size, total;
    POOLING_TYPE pooling_type;
    T *i_ptr ,*o_ptr, filter_size;
};

#endif