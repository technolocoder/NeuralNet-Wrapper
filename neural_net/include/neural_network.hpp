#ifndef __NEURAL_NETWORK_WRAPPER__
#define __NEURAL_NETWORK_WRAPPER__

#include "layer.hpp"
#include <random>

struct dim{
    int x,y,z;
};

TEMPLATE class neural_network{
public:
    neural_network();
    neural_network(int _sample_size, int _layer_count);

    void initialize(int _sample_size, int _layer_count);
    
    void init_from_neuralnet(int _sample_size ,neural_network net);
    void save_neural_net(const char *path);
    void load_neural_net(const char *path);

    void add_input_layer(int width, int height ,int depth, bool batch_normalized);
    void add_pooling_layer(int filter_width, int filter_height, int stride_width, int stride_height ,POOLING_TYPE pooling_type);
    void add_convolution_layer(int filter_width, int filter_height, int stride_width, int stride_height,int size, ACTIVATION_TYPES activation);
    void add_convolution_layer(int filter_width, int filter_height, int stride_width, int stride_height,int size, ACTIVATION_TYPES activation, T *filter);
    void add_flatten_layer();
    void add_fully_connected(int layer_size, ACTIVATION_TYPES activation);
    void add_softmax_layer();
    void construct_neuralnet();

    void set_sample_size(int _sample_size); // Must be below the initialized sample size

    T *feedforward(T *input);
    dim get_output_dimensions();

    void print_info();

    ~neural_network();
private:
    layer<T> *layers;
    
    int layer_count,sample_size,layer_index=1,total_size=0,fl_index=-1;
    bool init_layer = false, init_net = false;

    T * data;
    std::normal_distribution<T> dist = std::normal_distribution<T>((T)0.0,(T)1.0);
};

#endif