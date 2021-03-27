#ifndef __IL_WRAPPER__
#define __IL_WRAPPER__

#include "common.hpp"

TEMPLATE struct _IL{
    void init(int _width, int _height, int _depth, int _sample_size, bool _batch_normalized);
    void process();

    T *i_ptr, *o_ptr;
    int width,height,depth,sample_size,total;
    bool batch_normalized;
};

#endif 