#ifndef __LAYER_WRAPPER__
#define __LAYER_WRAPPER__

#include "common.hpp"

#include "IL.hpp"
#include "PL.hpp"
#include "CL.hpp"
#include "FL.hpp"

TEMPLATE struct layer{
    LAYER_TYPES type;
    union{
        _IL<T> IL;
        _PL<T> PL;
        _CL<T> CL;
        _FL<T> FL;
    };
};

#endif 