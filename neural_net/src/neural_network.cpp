#include "neural_network.hpp"
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;
random_device rd;
mt19937_64 engine(rd());
string activation_string[] = {"TANH","SIGMOID","RELU","LEAKY RELU","LINEAR"};

TEMPLATE neural_network<T>::neural_network() {}
TEMPLATE neural_network<T>::neural_network(int _sample_size ,int _layer_count){
    initialize(_sample_size,_layer_count);
}

TEMPLATE void neural_network<T>::initialize(int _sample_size, int _layer_count){
    layer_count = _layer_count;
    sample_size = _sample_size;

    layers = (layer<T>*)malloc(sizeof(layer<T>)*layer_count);
    init_layer = true;
}   

TEMPLATE void neural_network<T>::set_sample_size(int _sample_size){
    sample_size = _sample_size;
    layers[0].IL.sample_size = sample_size;
    for(int i = 1; i < layer_count; ++i){
        if(layers[i].type == POOLING_LAYER){
            layers[i].PL.sample_size = sample_size;
        }else if(layers[i].type == CONVOLUTION_LAYER){
            layers[i].CL.sample_size = sample_size;
        }else if(layers[i].type == FULLY_CONNECTED){
            layers[i].FC.sample_size = sample_size;
        }else{
            layers[i].SL.sample_size = sample_size;
        }
    }
}

TEMPLATE void neural_network<T>::add_input_layer(int width, int height, int depth, bool batch_normalized){
    layers[0].type = INPUT_LAYER;
    layers[0].IL.init(width,height,depth,sample_size,batch_normalized);
    total_size += layers[0].IL.total;
}

TEMPLATE void neural_network<T>::add_pooling_layer(int filter_width ,int filter_height ,int stride_width ,int stride_height ,POOLING_TYPE pooling_type){
    int type = layers[layer_index-1].type,width,height,depth;
    if(type == INPUT_LAYER){
        width = layers[layer_index-1].IL.width;
        height = layers[layer_index-1].IL.height;
        depth = layers[layer_index-1].IL.depth;
    }else if(type == POOLING_LAYER){
        width = layers[layer_index-1].PL.output_width;
        height = layers[layer_index-1].PL.output_height;
        depth = layers[layer_index-1].PL.depth;
    }else if(type == CONVOLUTION_LAYER){
        width = layers[layer_index-1].CL.output_width;
        height = layers[layer_index-1].CL.output_height;
        depth = layers[layer_index-1].CL.output_depth;
    }

    layers[layer_index].PL.init(width,height,filter_width,filter_height,stride_width,stride_height,depth,sample_size,pooling_type);
    layers[layer_index].type = POOLING_LAYER;
    total_size += layers[layer_index].PL.total;
    ++layer_index;
}

TEMPLATE void neural_network<T>::add_convolution_layer(int filter_width, int filter_height, int stride_width, int stride_height,int size, ACTIVATION_TYPES activation){
    int type = layers[layer_index-1].type,width,height,depth;
    if(type == INPUT_LAYER){
        width = layers[layer_index-1].IL.width;
        height = layers[layer_index-1].IL.height;
        depth = layers[layer_index-1].IL.depth;
    }else if(type == POOLING_LAYER){
        width = layers[layer_index-1].PL.output_width;
        height = layers[layer_index-1].PL.output_height;
        depth = layers[layer_index-1].PL.depth;
    }else if(type == CONVOLUTION_LAYER){
        width = layers[layer_index-1].CL.output_width;
        height = layers[layer_index-1].CL.output_height;
        depth = layers[layer_index-1].CL.output_depth;
    }
    layers[layer_index].CL.init(width,height,filter_width,filter_height,stride_width,stride_height,depth,size,sample_size,activation);
    layers[layer_index].type = CONVOLUTION_LAYER;
    total_size += layers[layer_index].CL.total;
    ++layer_index;
}

TEMPLATE void neural_network<T>::add_convolution_layer(int filter_width, int filter_height, int stride_width, int stride_height,int size, ACTIVATION_TYPES activation, T *filter){
    int type = layers[layer_index-1].type,width,height,depth;
    if(type == INPUT_LAYER){
        width = layers[layer_index-1].IL.width;
        height = layers[layer_index-1].IL.height;
        depth = layers[layer_index-1].IL.depth;
    }else if(type == POOLING_LAYER){
        width = layers[layer_index-1].PL.output_width;
        height = layers[layer_index-1].PL.output_height;
        depth = layers[layer_index-1].PL.depth;
    }else if(type == CONVOLUTION_LAYER){
        width = layers[layer_index-1].CL.output_width;
        height = layers[layer_index-1].CL.output_height;
        depth = layers[layer_index-1].CL.output_depth;
    }
    layers[layer_index].CL.init(width,height,filter_width,filter_height,stride_width,stride_height,depth,size,sample_size,activation);
    layers[layer_index].type = CONVOLUTION_LAYER;
    layers[layer_index].CL.temp_ptr = filter;
    layers[layer_index].CL.preset = true;
    total_size += layers[layer_index].CL.total;
    ++layer_index;
}

TEMPLATE void neural_network<T>::add_flatten_layer(){
    layers[layer_index].type = FLATTEN_LAYER;
    int total;
    if(layers[layer_index-1].type == CONVOLUTION_LAYER){
        total = layers[layer_index-1].CL.output_size;
    }else if(layers[layer_index-1].type == POOLING_LAYER){
        total = layers[layer_index-1].PL.total;
    }else if(layers[layer_index-1].type == INPUT_LAYER){
        total = layers[layer_index-1].IL.width*layers[layer_index-1].IL.height*layers[layer_index-1].IL.depth*sample_size;
    }
    layers[layer_index].FL.total = total;
    layers[layer_index].FL.output_size = total/sample_size;
    
    fl_index = layer_index;
    ++layer_index;
}

TEMPLATE void neural_network<T>::add_fully_connected(int layer_size ,ACTIVATION_TYPES activation){
    layers[layer_index].type = FULLY_CONNECTED;
    int input_size;
    if(layers[layer_index-1].type == FULLY_CONNECTED){
        input_size = layers[layer_index-1].FC.layer_size;
    }else{
        input_size = layers[layer_index-1].FL.output_size;
    }
    layers[layer_index].FC.init(input_size,layer_size,sample_size,activation);
    total_size += layers[layer_index].FC.total;
    ++layer_index;
}

TEMPLATE void neural_network<T>::add_softmax_layer(){
    layers[layer_index].type = SOFTMAX_LAYER;
    int input_size;
    if(layers[layer_index-1].type == FULLY_CONNECTED){
        input_size = layers[layer_index-1].FC.layer_size;
    }else{
        input_size = layers[layer_index-1].FL.output_size;
    }
    layers[layer_index].SL.init(input_size,sample_size);
    total_size += layers[layer_index].SL.total;
    ++layer_index;
}

TEMPLATE void neural_network<T>::construct_neuralnet(){
    data = (T*)malloc(total_size*sizeof(T));
    init_net=true;

    int offset = 0;
    layers[0].IL.i_ptr = NULL;
    if(layers[0].IL.batch_normalized){
        layers[0].IL.o_ptr = data;
        offset += layers[0].IL.total;
    }else{
        layers[0].IL.o_ptr = NULL;
    }

    T *prev_ptr = layers[0].IL.o_ptr;
    if(fl_index>0){
        for(int i = 1; i < fl_index; ++i){
            if(layers[i].type == POOLING_LAYER){
                layers[i].PL.i_ptr = prev_ptr;
                layers[i].PL.o_ptr = data+offset;
                offset += layers[i].PL.total;
                prev_ptr = layers[i].PL.o_ptr;
            }else if(layers[i].type == CONVOLUTION_LAYER){
                layers[i].CL.i_ptr = prev_ptr;
                layers[i].CL.k_ptr = data+offset;
                layers[i].CL.b_ptr = data+(offset+layers[i].CL.kernel_size);
                layers[i].CL.o_ptr = data+(offset+layers[i].CL.kernel_size+layers[i].CL.output_depth);
                memset(layers[i].CL.o_ptr,0,sizeof(T)*layers[i].CL.output_depth);
                if(!layers[i].CL.preset){
                    for(int j = 0; j < layers[i].CL.kernel_size; ++j) layers[i].CL.k_ptr[j] = dist(engine);
                }else{
                    memcpy(layers[i].CL.k_ptr,layers[i].CL.temp_ptr,sizeof(T)*layers[i].CL.kernel_size);
                }
                offset += layers[i].CL.total;
                prev_ptr = layers[i].CL.o_ptr;
            }
        }
        layers[fl_index].FL.ptr = prev_ptr;
        for(int i = fl_index+1; i < layer_count; ++i){
            if(layers[i].type == FULLY_CONNECTED){
                layers[i].FC.i_ptr = prev_ptr;
                layers[i].FC.w_ptr = data+offset;
                for(int j = 0; j < layers[i].FC.weight_count; ++j) layers[i].FC.w_ptr[j] = dist(engine);
                layers[i].FC.b_ptr = data+(offset+layers[i].FC.weight_count);
                memset(layers[i].FC.b_ptr,0,sizeof(T)*layers[i].FC.layer_size);
                layers[i].FC.o_ptr = data+(offset+layers[i].FC.weight_count+layers[i].FC.layer_size);
                offset += layers[i].FC.total;
                prev_ptr = layers[i].FC.o_ptr;
            }else{
                layers[i].SL.i_ptr = prev_ptr;
                layers[i].SL.o_ptr = data+offset;
                prev_ptr = layers[i].SL.o_ptr;
                offset += layers[i].SL.total;
            }
        }
    }else{
        for(int i = 1; i < layer_count; ++i){
            if(layers[i].type == POOLING_LAYER){
                layers[i].PL.i_ptr = prev_ptr;
                layers[i].PL.o_ptr = data+offset;
                offset += layers[i].PL.total;
                prev_ptr = layers[i].PL.o_ptr;
            }else if(layers[i].type == CONVOLUTION_LAYER){
                layers[i].CL.i_ptr = prev_ptr;
                layers[i].CL.k_ptr = data+offset;
                layers[i].CL.b_ptr = data+(offset+layers[i].CL.kernel_size);
                layers[i].CL.o_ptr = data+(offset+layers[i].CL.kernel_size+layers[i].CL.output_depth);
                memset(layers[i].CL.b_ptr,0,sizeof(T)*layers[i].CL.output_depth);
                if(!layers[i].CL.preset){
                    for(int j = 0; j < layers[i].CL.kernel_size; ++j) layers[i].CL.k_ptr[j] = dist(engine);
                }else{
                    memcpy(layers[i].CL.k_ptr,layers[i].CL.temp_ptr,sizeof(T)*layers[i].CL.kernel_size);
                }
                offset += layers[i].CL.total;
                prev_ptr = layers[i].CL.o_ptr;
            }
        }        
    }
}

TEMPLATE T* neural_network<T>::feedforward(T *input){
    layers[0].IL.i_ptr = input;
    layers[0].IL.process();
    if(layers[1].type == POOLING_LAYER){
        layers[1].PL.i_ptr = layers[0].IL.o_ptr;
        layers[1].PL.process();
    }else if(layers[1].type == CONVOLUTION_LAYER){
        layers[1].CL.i_ptr = layers[0].IL.o_ptr;
        layers[1].CL.process();
    }else{
        layers[1].FL.ptr = layers[0].IL.o_ptr;
    }

    if(fl_index < 0){
        for(int i = 2; i < layer_count; ++i){
            if(layers[i].type == POOLING_LAYER) layers[i].PL.process();
            else if(layers[i].type == CONVOLUTION_LAYER) layers[i].CL.process();
        }
        return layers[layer_count-1].type==POOLING_LAYER?layers[layer_count-1].PL.o_ptr:layers[layer_count-1].type==CONVOLUTION_LAYER?layers[layer_count-1].CL.o_ptr:layers[layer_count-1].IL.o_ptr;
    }else{
        for(int i = 2; i < fl_index; ++i){
            if(layers[i].type == POOLING_LAYER) layers[i].PL.process();
            else if(layers[i].type == CONVOLUTION_LAYER) layers[i].CL.process();
        }
        if(layer_count-fl_index>1){
            if(layers[fl_index+1].type == FULLY_CONNECTED){
                layers[fl_index+1].FC.i_ptr = layers[fl_index].FL.ptr;
            }else{
                layers[fl_index+1].SL.i_ptr = layers[fl_index].FL.ptr;
            }
            for(int i = fl_index+1; i < layer_count; ++i){
                if(layers[i].type == FULLY_CONNECTED){
                    layers[i].FC.process();
                }else{
                    layers[i].SL.process();
                }
            }
        }
        return layers[layer_count-1].type==SOFTMAX_LAYER?layers[layer_count-1].SL.o_ptr:layers[layer_count-1].type==FULLY_CONNECTED?layers[layer_count-1].FC.o_ptr:layers[layer_count-1].FL.ptr;
    }
}

TEMPLATE dim neural_network<T>::get_output_dimensions(){
    if(layers[layer_count-1].type == POOLING_LAYER){
        return {layers[layer_count-1].PL.output_width,layers[layer_count-1].PL.output_height,layers[layer_count-1].PL.depth};
    }else if(layers[layer_count-1].type == CONVOLUTION_LAYER){
        return {layers[layer_count-1].CL.output_width,layers[layer_count-1].CL.output_height,layers[layer_count-1].CL.output_depth};
    }else if(layers[layer_count-1].type == FLATTEN_LAYER){
        return {layers[layer_count-1].FL.output_size,1,1};
    }else if(layers[layer_count-1].type == FULLY_CONNECTED){
        return {layers[layer_count-1].FC.layer_size,1,1};
    }else if(layers[layer_count-1].type == SOFTMAX_LAYER){
        return {layers[layer_count-1].SL.layer_size,1,1};
    }
    return {layers[layer_count-1].IL.width,layers[layer_count-1].IL.height,layers[layer_count-1].IL.depth};
}


TEMPLATE void neural_network<T>::print_info(){
    cout << "Sample Size: " << sample_size << "\nLayer Count: " << layer_count << "\nInput Layer: 0\nWidth: " << layers[0].IL.width << "\nHeight: " << layers[0].IL.height << "\nDepth: " << layers[0].IL.depth << "\nBatch Normalized: " << (layers[0].IL.batch_normalized?"YES":"NO") << '\n';
    cout << "-------------------------\n";
    for(int i = 1; i < layer_count; ++i){
        layer l = layers[i];
        if(layers[i].type == CONVOLUTION_LAYER){
            cout << "Convolution Layer: " << i << "\nInput Width: " << l.CL.input_width << "\nInput Height: " << l.CL.input_height << "\nInput Depth: " << l.CL.input_depth << "\nKernel Width: " << l.CL.filter_width << "\nKernel Height: " << l.CL.filter_height << "\nStride Width: " << l.CL.stride_width << "\nStride Height: " << l.CL.stride_height << "\nOutput Width: " << l.CL.output_width << "\nOutput Height: " << l.CL.output_height << "\nOutput Depth: " << l.CL.output_depth << "\nActivation: " << activation_string[l.CL.activation] << '\n';
        }else if(layers[i].type == POOLING_LAYER){
            cout << "Pooling Layer: " << i << "\nInput Width: " << l.PL.input_width << "\nInput Height: " << l.PL.input_height << "\nDepth: " << l.PL.depth << "\nKernel Width: " << l.PL.filter_width << "\nKernel Height: " << l.PL.filter_height << "\nStride Width: " << l.PL.stride_width << "\nStride Height: " << l.PL.stride_height << "\nOutput Width: " << l.PL.output_width << "\nOutput Height: " << l.PL.output_height << "\nPooling Type: " << (l.PL.pooling_type==MAX_POOL?"Max Pool\n":"Avg Pool\n");
        }else if(layers[i].type == FLATTEN_LAYER){
            cout << "Flatten Layer: " << i << "\nTotal Size: " << l.FL.total << '\n';
        }else if(layers[i].type == FULLY_CONNECTED){
            cout << "Fully Connected Layer: " << i << "\nInput Size: " << l.FC.input_size << "\nLayer Size: " << l.FC.layer_size << "\nWeight Count: " << l.FC.weight_count << "\nActivation: " << activation_string[l.FC.activation] << '\n';
        }else{
            cout << "Softmax Layer: " << i << "\nLayer Size: " << l.SL.layer_size << '\n';
        }
        cout << "-------------------------\n";
    }
}

TEMPLATE neural_network<T>::~neural_network(){
    if(init_layer) free(layers);
    if(init_net) free(data);
}

template class neural_network<float>;
template class neural_network<double>;